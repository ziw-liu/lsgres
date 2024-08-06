use anyhow::{bail, Context, Result};
use clap::Parser;
use colored::{ColoredString, Colorize};
use once_cell::sync::Lazy;
use regex::Regex;
use serde::Deserialize;
use tabled::{Table, Tabled};

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
    fn from_node(node: &Node) -> Result<Self> {
        let gres_total = GresStatus::from_str(&node.gres)?;
        let gres_used = GresStatus::from_str(&node.gres_used)?;
        let idle_count = gres_total.count - gres_used.count;
        let used_print = repeat_colored_char('u', gres_used.count, "red");
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
            gres_status: format!("{}{}", used_print, idle_print),
            state: state_colored,
        })
    }
}

fn query_nodes() -> Result<SlurmNodes> {
    let scontrol = "scontrol";
    let args = ["show", "nodes", "--json"];
    let output = std::process::Command::new(scontrol).args(args).output()?;
    if output.status.success() {
        let output_str = std::str::from_utf8(&output.stdout)?;
        let result: SlurmNodes = serde_json::from_str(output_str)?;
        Ok(result)
    } else {
        let error_msg = String::from_utf8(output.stderr)?;
        bail!("Scontrol failed: {}", &error_msg)
    }
}

#[derive(Parser)]
#[command(version, about)]
struct Cli {
    // Name of the general resource
    gres: String,

    // Only show one partition
    #[arg(short, long)]
    partition: Option<String>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let slurm_nodes = query_nodes()?;
    let matched: Result<Vec<TableNode>> = slurm_nodes
        .nodes
        .iter()
        .filter(|&node| {
            let mut gres_matched = node.gres.contains(&cli.gres);
            if let Some(ref partition) = cli.partition {
                gres_matched &= node.partitions.contains(partition)
            }
            gres_matched
        })
        .map(TableNode::from_node)
        .collect();
    let tabled_nodes = matched?;
    let table = Table::new(tabled_nodes).to_string();
    println!("{}", table);
    Ok(())
}
