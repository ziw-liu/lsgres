use anyhow::{bail, Context, Result};
use clap::{Parser, ValueEnum};
use colored::{ColoredString, Colorize};
use once_cell::sync::Lazy;
use regex::Regex;
use serde::Deserialize;
use tabled::{settings::Style, Table, Tabled};

#[derive(Deserialize, Debug)]
struct SlurmJobs {
    jobs: Vec<Job>,
}

#[derive(Deserialize, Debug, Clone)]
struct Job {
    partition: String,
    nodes: String,
    #[serde(default)]
    gres_detail: Vec<String>,
}

#[derive(Debug, PartialEq)]
struct GpuAllocation {
    node: String,
    gpus: u32,
}

/// Parses a "gres_detail" string to extract GPU allocation details.
/// Returns an Option containing a GpuAllocation if parsing is successful.
fn parse_gpu_allocation(gres_str: &str, node: &str) -> Option<GpuAllocation> {
    if !gres_str.starts_with("gpu") {
        return None;
    }

    let parts: Vec<&str> = gres_str.split(':').collect();
    if parts.len() < 3 {
        return None;
    }

    // Check if there's an IDX specification in parentheses
    if let Some(idx_start) = gres_str.find("(IDX:") {
        if let Some(idx_end) = gres_str[idx_start..].find(')') {
            let idx_spec = &gres_str[idx_start + 5..idx_start + idx_end];
            let gpu_count = count_gpu_indices(idx_spec);
            if gpu_count > 0 {
                return Some(GpuAllocation {
                    node: node.to_string(),
                    gpus: gpu_count,
                });
            }
        }
    }

    // Fallback to the original method if no IDX specification
    let count_str: String = parts[2].chars().take_while(|c| c.is_digit(10)).collect();

    // Use `and_then` to chain the parsing and filtering.
    count_str.parse::<u32>().ok().and_then(|gpu_count| {
        if gpu_count > 0 {
            Some(GpuAllocation {
                node: node.to_string(),
                gpus: gpu_count,
            })
        } else {
            None
        }
    })
}

/// Counts the number of GPU indices specified in an IDX string like "0,2-7" or "0-3"
fn count_gpu_indices(idx_spec: &str) -> u32 {
    let mut count = 0;
    for part in idx_spec.split(',') {
        if let Some(dash_pos) = part.find('-') {
            // Range like "2-7"
            let start = part[..dash_pos].parse::<u32>().unwrap_or(0);
            let end = part[dash_pos + 1..].parse::<u32>().unwrap_or(0);
            if end >= start {
                count += end - start + 1;
            }
        } else {
            // Single index like "0"
            if part.parse::<u32>().is_ok() {
                count += 1;
            }
        }
    }
    count
}

/// Filters jobs to find those in the "preempted" partition and extracts their GPU allocations.
fn process_preempted_jobs(jobs: &[Job]) -> Vec<GpuAllocation> {
    jobs.iter()
        .filter(|job| job.partition == "preempted")
        .flat_map(|job| {
            job.gres_detail
                .iter()
                .filter_map(move |gres| parse_gpu_allocation(gres, &job.nodes))
        })
        .collect()
}

#[derive(Deserialize, Debug)]
struct Node {
    hostname: String,
    state: Vec<String>,
    partitions: Vec<String>,
    cpus: usize,
    alloc_idle_cpus: usize,
    real_memory: usize,
    alloc_memory: usize,
    gres: String,
    gres_used: String,
}

#[derive(Deserialize, Debug)]
struct SlurmNodes {
    nodes: Vec<Node>,
}

struct GresStatus {
    model: String,
    count: usize,
}

impl GresStatus {
    fn from_str(s: &str) -> Result<Self> {
        if s.is_empty() || s == "(null)" {
            return Ok(Self {
                model: "".to_string(),
                count: 0,
            });
        }

        static RE: Lazy<Regex> =
            Lazy::new(|| Regex::new(r"(?P<model>\w+:\w+):(?P<count>\d+)").unwrap());
        let caps = RE.captures(s).context("Matching Gres status failed!")?;
        Ok(Self {
            model: caps["model"].to_string(),
            count: caps["count"].parse::<usize>()?,
        })
    }
}

fn repeat_colored_char(character: char, number: usize, color: &str) -> ColoredString {
    std::iter::repeat(character)
        .take(number)
        .collect::<String>()
        .color(color)
}

fn format_ratio(used: usize, total: usize) -> String {
    format!("{}/{}", used, total)
}

#[derive(Tabled)]
struct TableNode {
    hostname: String,
    cpus_available: String,
    memory_available: String,
    gres: String,
    gres_status: String,
    state: String,
}

impl TableNode {
    fn from_node(node: &Node, preempted_gpus: &[GpuAllocation]) -> Result<Self> {
        let gres_total = GresStatus::from_str(&node.gres)?;
        let gres_used = GresStatus::from_str(&node.gres_used)?;
        let idle_count = gres_total.count - gres_used.count;

        // Check if this node has preempted GPUs
        let preempted_count = preempted_gpus
            .iter()
            .find(|gpu| gpu.node == node.hostname)
            .map(|gpu| gpu.gpus as usize)
            .unwrap_or(0);

        let regular_used_count = gres_used.count.saturating_sub(preempted_count);

        let regular_used_print = repeat_colored_char('u', regular_used_count, "red");
        let preempted_print = repeat_colored_char('p', preempted_count, "yellow");
        let idle_print = repeat_colored_char('i', idle_count, "green");

        let state_colored = node
            .state
            .iter()
            .map(|s| match s.as_str() {
                "IDLE" => s.green().to_string(),
                "MIXED" => s.blue().to_string(),
                "ALLOCATED" => s.magenta().to_string(),
                "DRAIN" => s.yellow().to_string(),
                "DOWN" => s.red().to_string(),
                _ => s.to_owned(),
            })
            .collect::<Vec<String>>()
            .join(",");
        Ok(Self {
            hostname: node.hostname.clone(),
            cpus_available: format_ratio(node.alloc_idle_cpus, node.cpus),
            memory_available: format_ratio(
                (node.real_memory - node.alloc_memory) / 1000,
                node.real_memory / 1000,
            ) + "G",
            gres: gres_total.model,
            gres_status: format!("{}{}{}", regular_used_print, preempted_print, idle_print),
            state: state_colored,
        })
    }
}

fn run_scontrol_command<T>(args: &[&str]) -> Result<T>
where
    T: for<'de> Deserialize<'de>,
{
    let output = std::process::Command::new("scontrol").args(args).output()?;
    if output.status.success() {
        let output_str = std::str::from_utf8(&output.stdout)?;
        let result: T = serde_json::from_str(output_str)?;
        Ok(result)
    } else {
        let error_msg = String::from_utf8(output.stderr)?;
        bail!("Scontrol failed: {}", &error_msg)
    }
}

fn query_nodes() -> Result<SlurmNodes> {
    run_scontrol_command(&["show", "nodes", "--json"])
}

fn query_jobs() -> Result<SlurmJobs> {
    run_scontrol_command(&["show", "job", "--json"])
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum TableStyle {
    Markdown,
    Ascii,
    Modern,
}

fn apply_style_to_table(style: Option<TableStyle>, table: &mut Table) -> &Table {
    match style {
        Some(TableStyle::Markdown) | None => table.with(Style::markdown()),
        Some(TableStyle::Ascii) => table.with(Style::ascii()),
        Some(TableStyle::Modern) => table.with(Style::modern()),
    }
}

#[derive(Parser)]
#[command(
    version,
    about = "List generic resource (GRES) in a Slurm cluster by node"
)]
struct Cli {
    /// Name of the GRES, e.g. "gpu", "h100", "a6000". Leave empty to show all nodes
    gres: Option<String>,

    /// Select which partition to show, e.g. "gpu", "interactive"
    #[arg(short, long)]
    partition: Option<String>,

    /// Style of the printed table, by default "markdown"
    #[arg(short, long, value_enum)]
    style: Option<TableStyle>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let slurm_nodes = query_nodes()?;
    let slurm_jobs = query_jobs()?;
    let preempted_gpus = process_preempted_jobs(&slurm_jobs.jobs);

    let matched: Result<Vec<TableNode>> = slurm_nodes
        .nodes
        .iter()
        .filter(|&node| {
            let mut gres_matched = cli.gres.is_none()
                || node
                    .gres
                    .contains(cli.gres.as_ref().unwrap_or(&"".into()).as_str());
            if let Some(ref partition) = cli.partition {
                gres_matched &= node.partitions.contains(partition)
            }
            gres_matched
        })
        .map(|node| TableNode::from_node(node, &preempted_gpus))
        .collect();
    let tabled_nodes = matched?;
    let mut table = Table::new(tabled_nodes);
    apply_style_to_table(cli.style, &mut table);
    println!("{}", table);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_gpu_allocation_success() {
        let gres = "gpu:a40:1(IDX:0)";
        let node = "gpu-sm01-13";
        let expected = Some(GpuAllocation {
            node: node.to_string(),
            gpus: 1,
        });
        assert_eq!(parse_gpu_allocation(gres, node), expected);
    }

    #[test]
    fn test_parse_gpu_allocation_multi_digit() {
        let gres = "gpu:a40:16(IDX:0-15)";
        let node = "gpu-sm01-14";
        let expected = Some(GpuAllocation {
            node: node.to_string(),
            gpus: 16,
        });
        assert_eq!(parse_gpu_allocation(gres, node), expected);
    }

    #[test]
    fn test_parse_gpu_allocation_failure() {
        let gres = "cpu:2";
        let node = "cpu-node-1";
        assert_eq!(parse_gpu_allocation(gres, node), None);
    }

    #[test]
    fn test_process_preempted_jobs() {
        let jobs = vec![
            Job {
                partition: "cpu".to_string(),
                nodes: "cpu-a-1".to_string(),
                gres_detail: vec![],
            },
            Job {
                partition: "gpu".to_string(),
                nodes: "gpu-a-1".to_string(),
                gres_detail: vec!["gpu:a100:1(IDX:3)".to_string()],
            },
            Job {
                partition: "interactive".to_string(),
                nodes: "gpu-sm01-14".to_string(),
                gres_detail: vec!["gpu:a40:1(IDX:0)".to_string()],
            },
            Job {
                partition: "preempted".to_string(),
                nodes: "gpu-sm01-13".to_string(),
                gres_detail: vec!["gpu:a40:1(IDX:0)".to_string()],
            },
            Job {
                partition: "preempted".to_string(),
                nodes: "gpu-f-6".to_string(),
                gres_detail: vec!["gpu:h100:4(IDX:0-3)".to_string()],
            },
            Job {
                partition: "preempted".to_string(),
                nodes: "gpu-h-2".to_string(),
                gres_detail: vec!["gpu:h200:7(IDX:0,2-7)".to_string()],
            },
            Job {
                partition: "preempted".to_string(),
                nodes: "another-node".to_string(),
                gres_detail: vec![],
            },
        ];

        let expected = vec![
            GpuAllocation {
                node: "gpu-sm01-13".to_string(),
                gpus: 1,
            },
            GpuAllocation {
                node: "gpu-f-6".to_string(),
                gpus: 4,
            },
            GpuAllocation {
                node: "gpu-h-2".to_string(),
                gpus: 7,
            },
        ];

        let result = process_preempted_jobs(&jobs);
        assert_eq!(result, expected);
    }
}
