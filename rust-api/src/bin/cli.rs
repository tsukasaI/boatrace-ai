//! Boatrace CLI - Command-line interface for boat race predictions

use clap::{Parser, Subcommand};
use colored::Colorize;

use boatrace::core::kelly::KellyCalculator;
use boatrace::predictor::FallbackPredictor;

#[derive(Parser)]
#[command(name = "boatrace")]
#[command(author, version, about = "Boat race prediction CLI", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Run in interactive mode
    #[arg(short, long)]
    interactive: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Predict a single race
    Predict {
        /// Race date (YYYYMMDD format)
        #[arg(short, long)]
        date: u32,

        /// Stadium code (1-24)
        #[arg(short, long)]
        stadium: u8,

        /// Race number (1-12)
        #[arg(short, long)]
        race: u8,

        /// Include trifecta predictions
        #[arg(long)]
        trifecta: bool,

        /// Bankroll amount for Kelly sizing
        #[arg(long, default_value = "100000")]
        bankroll: i64,
    },

    /// List available races for a date
    List {
        /// Race date (YYYYMMDD format)
        #[arg(short, long)]
        date: u32,
    },

    /// Run backtesting simulation
    Backtest {
        /// EV threshold for betting
        #[arg(long, default_value = "1.0")]
        threshold: f64,

        /// Use synthetic odds instead of real odds
        #[arg(long)]
        synthetic_odds: bool,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    println!("{}", "Boatrace CLI v0.2.0".cyan().bold());
    println!();

    if cli.interactive {
        run_interactive()?;
    } else if let Some(command) = cli.command {
        match command {
            Commands::Predict {
                date,
                stadium,
                race,
                trifecta,
                bankroll,
            } => {
                predict_race(date, stadium, race, trifecta, bankroll)?;
            }
            Commands::List { date } => {
                list_races(date)?;
            }
            Commands::Backtest {
                threshold,
                synthetic_odds,
            } => {
                run_backtest(threshold, synthetic_odds)?;
            }
        }
    } else {
        println!("Use --help for usage information or --interactive for interactive mode.");
    }

    Ok(())
}

fn predict_race(
    date: u32,
    stadium: u8,
    race: u8,
    trifecta: bool,
    bankroll: i64,
) -> anyhow::Result<()> {
    println!(
        "{}: {} / Stadium {} / Race {}",
        "Predicting".green(),
        date,
        stadium,
        race
    );
    println!("Trifecta: {}", if trifecta { "Yes" } else { "No" });
    println!("Bankroll: {}", bankroll);
    println!();

    // TODO: Load race data from CSV
    // TODO: Run prediction
    // TODO: Display results

    // Demo with fallback predictor
    let _predictor = FallbackPredictor::new();
    let calc = KellyCalculator::with_defaults(bankroll);

    // Example bet sizing
    let sizing = calc.calculate_single(0.25, 5.0);
    println!("{}", "Example bet sizing:".yellow());
    println!("  Probability: 25%");
    println!("  Odds: 5.0x");
    println!("  Expected Value: {:.2}", sizing.expected_value);
    println!("  Kelly fraction: {:.4}", sizing.kelly_fraction);
    println!("  Recommended stake: {}", sizing.stake);

    println!();
    println!(
        "{}",
        "Note: Full prediction requires CSV data loading (Phase 2)".dimmed()
    );

    Ok(())
}

fn list_races(date: u32) -> anyhow::Result<()> {
    println!("{}: {}", "Listing races for".green(), date);
    println!();

    // TODO: Load race data from CSV and display available races
    println!(
        "{}",
        "Note: Race listing requires CSV data loading (Phase 2)".dimmed()
    );

    Ok(())
}

fn run_backtest(threshold: f64, synthetic_odds: bool) -> anyhow::Result<()> {
    println!("{}", "Running backtest...".green());
    println!("EV Threshold: {}", threshold);
    println!(
        "Synthetic odds: {}",
        if synthetic_odds { "Yes" } else { "No" }
    );
    println!();

    // TODO: Implement backtesting
    println!(
        "{}",
        "Note: Backtesting requires implementation (Phase 5)".dimmed()
    );

    Ok(())
}

fn run_interactive() -> anyhow::Result<()> {
    println!("{}", "Interactive mode".green().bold());
    println!("Type 'help' for available commands, 'quit' to exit.");
    println!();

    // TODO: Implement interactive REPL using dialoguer
    println!(
        "{}",
        "Note: Interactive mode requires dialoguer integration (Phase 3)".dimmed()
    );

    Ok(())
}
