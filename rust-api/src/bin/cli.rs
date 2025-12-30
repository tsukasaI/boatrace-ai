//! Boatrace CLI - Command-line interface for boat race predictions

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use colored::Colorize;
use dialoguer::{theme::ColorfulTheme, Input, Select};
use indicatif::{ProgressBar, ProgressStyle};
use std::path::{Path, PathBuf};

use boatrace::core::kelly::KellyCalculator;
use boatrace::data::{load_exacta_odds, load_trifecta_odds, RaceData};
use boatrace::predictor::FallbackPredictor;
use boatrace::{ExactaPrediction, RacerEntry, TrifectaPrediction};

/// Default data directory (relative to project root)
const DEFAULT_DATA_DIR: &str = "data/processed";
const DEFAULT_ODDS_DIR: &str = "data/odds";

#[derive(Parser)]
#[command(name = "boatrace")]
#[command(author, version, about = "Boat race prediction CLI", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Run in interactive mode
    #[arg(short, long)]
    interactive: bool,

    /// Path to processed data directory
    #[arg(long, default_value = DEFAULT_DATA_DIR)]
    data_dir: PathBuf,

    /// Path to odds directory
    #[arg(long, default_value = DEFAULT_ODDS_DIR)]
    odds_dir: PathBuf,
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

        /// Kelly multiplier (0.25 = quarter Kelly)
        #[arg(long, default_value = "0.25")]
        kelly: f64,

        /// EV threshold for betting recommendations
        #[arg(long, default_value = "1.0")]
        threshold: f64,

        /// Number of top predictions to show
        #[arg(long, default_value = "10")]
        top: usize,
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

fn main() -> Result<()> {
    let cli = Cli::parse();

    println!("{}", "Boatrace CLI v0.2.0".cyan().bold());
    println!();

    if cli.interactive {
        run_interactive(&cli.data_dir, &cli.odds_dir)?;
    } else if let Some(command) = cli.command {
        match command {
            Commands::Predict {
                date,
                stadium,
                race,
                trifecta,
                bankroll,
                kelly,
                threshold,
                top,
            } => {
                predict_race(
                    &cli.data_dir,
                    &cli.odds_dir,
                    date,
                    stadium,
                    race,
                    trifecta,
                    bankroll,
                    kelly,
                    threshold,
                    top,
                )?;
            }
            Commands::List { date } => {
                list_races(&cli.data_dir, date)?;
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

/// Stadium code to name mapping
fn stadium_name(code: u8) -> &'static str {
    match code {
        1 => "桐生",
        2 => "戸田",
        3 => "江戸川",
        4 => "平和島",
        5 => "多摩川",
        6 => "浜名湖",
        7 => "蒲郡",
        8 => "常滑",
        9 => "津",
        10 => "三国",
        11 => "びわこ",
        12 => "住之江",
        13 => "尼崎",
        14 => "鳴門",
        15 => "丸亀",
        16 => "児島",
        17 => "宮島",
        18 => "徳山",
        19 => "下関",
        20 => "若松",
        21 => "芦屋",
        22 => "福岡",
        23 => "唐津",
        24 => "大村",
        _ => "不明",
    }
}

#[allow(clippy::too_many_arguments)]
fn predict_race(
    data_dir: &Path,
    odds_dir: &Path,
    date: u32,
    stadium: u8,
    race: u8,
    trifecta: bool,
    bankroll: i64,
    kelly_mult: f64,
    threshold: f64,
    top: usize,
) -> Result<()> {
    println!(
        "{}: {} ({}) / {} / {}R",
        "Predicting".green(),
        date,
        format_date(date),
        stadium_name(stadium),
        race
    );
    println!();

    // Load race data
    let csv_path = data_dir.join("programs_entries.csv");
    let race_data = RaceData::load(&csv_path)
        .with_context(|| format!("Failed to load CSV from {:?}", csv_path))?;

    let entries = race_data
        .get_race(date, stadium, race)
        .with_context(|| format!("Failed to get race {}/{}/{}", date, stadium, race))?;

    if entries.is_empty() {
        println!("{}", "No entries found for this race.".red());
        return Ok(());
    }

    // Display race entries
    println!("{}", "出走表 (Race Entries):".yellow().bold());
    println!(
        "{:>4} {:>6} {:<12} {:>4} {:>3} {:>6} {:>6} {:>6} {:>6}",
        "艇番", "登番", "選手名", "級", "歳", "全国勝", "当地勝", "モ2率", "ボ2率"
    );
    println!("{}", "-".repeat(70));

    let racer_entries: Vec<RacerEntry> = entries.iter().map(|e| e.to_racer_entry()).collect();

    for entry in &racer_entries {
        println!(
            "{:>4} {:>6} {:<12} {:>4} {:>3} {:>6.2} {:>6.2} {:>6.2} {:>6.2}",
            entry.boat_no,
            entry.racer_id,
            truncate_name(&entry.racer_name, 12),
            entry.racer_class,
            entry.age,
            entry.national_win_rate,
            entry.local_win_rate,
            entry.motor_in2_rate,
            entry.boat_in2_rate
        );
    }
    println!();

    // Run prediction
    let predictor = FallbackPredictor::new();
    let position_probs = predictor.predict_positions(&racer_entries);

    // Display position probabilities
    println!("{}", "着順予想 (Position Probabilities):".yellow().bold());
    println!(
        "{:>4} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "艇番", "1着", "2着", "3着", "4着", "5着", "6着"
    );
    println!("{}", "-".repeat(60));

    for prob in &position_probs {
        println!(
            "{:>4} {:>7.1}% {:>7.1}% {:>7.1}% {:>7.1}% {:>7.1}% {:>7.1}%",
            prob.boat_no,
            prob.probs[0] * 100.0,
            prob.probs[1] * 100.0,
            prob.probs[2] * 100.0,
            prob.probs[3] * 100.0,
            prob.probs[4] * 100.0,
            prob.probs[5] * 100.0
        );
    }
    println!();

    // Load real odds if available
    let exacta_odds = load_exacta_odds(odds_dir, date, stadium, race);
    let trifecta_odds_map = if trifecta {
        load_trifecta_odds(odds_dir, date, stadium, race)
    } else {
        None
    };

    // Calculate exacta predictions
    let exacta_probs = predictor.calculate_exacta_probs(&position_probs);
    let kelly = KellyCalculator::new(bankroll, kelly_mult, 100, 0.10, 0.30);

    // Display exacta predictions with EV
    println!(
        "{}",
        "2連単予想 (Exacta Predictions - Top):".yellow().bold()
    );

    let has_odds = exacta_odds.is_some();
    if has_odds {
        println!(
            "{:>8} {:>10} {:>8} {:>8} {:>10}",
            "組合せ", "確率", "オッズ", "EV", "推奨額"
        );
    } else {
        println!("{:>8} {:>10}", "組合せ", "確率");
        println!(
            "{}",
            "(オッズデータがありません。--odds-dir でディレクトリを指定してください)".dimmed()
        );
    }
    println!("{}", "-".repeat(50));

    // Add odds and calculate EV
    let mut exacta_with_ev: Vec<(ExactaPrediction, Option<f64>, f64, i64)> = exacta_probs
        .iter()
        .map(|pred| {
            let odds = exacta_odds
                .as_ref()
                .and_then(|o| o.get(&(pred.first, pred.second)).copied());
            let ev = odds.map(|o| pred.probability * o).unwrap_or(0.0);
            let stake = if let Some(o) = odds {
                kelly.calculate_single(pred.probability, o).stake
            } else {
                0
            };
            (pred.clone(), odds, ev, stake)
        })
        .collect();

    // Sort by EV descending
    exacta_with_ev.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    let mut bet_count = 0;
    for (pred, odds, ev, stake) in exacta_with_ev.iter().take(top) {
        let combo = format!("{}-{}", pred.first, pred.second);

        if has_odds {
            let odds_str = odds.map(|o| format!("{:.1}", o)).unwrap_or("-".to_string());
            let ev_str = if ev > &0.0 {
                format!("{:.3}", ev)
            } else {
                "-".to_string()
            };
            let stake_str = if *stake > 0 {
                format!("¥{}", stake)
            } else {
                "-".to_string()
            };

            let ev_color = if *ev >= threshold {
                bet_count += 1;
                ev_str.green()
            } else {
                ev_str.normal()
            };

            println!(
                "{:>8} {:>9.2}% {:>8} {:>8} {:>10}",
                combo,
                pred.probability * 100.0,
                odds_str,
                ev_color,
                stake_str
            );
        } else {
            println!("{:>8} {:>9.2}%", combo, pred.probability * 100.0);
        }
    }
    println!();

    if has_odds && bet_count > 0 {
        println!(
            "{} EV ≥ {:.1} のベット候補: {}件",
            "→".green(),
            threshold,
            bet_count
        );
        println!();
    }

    // Trifecta predictions
    if trifecta {
        println!(
            "{}",
            "3連単予想 (Trifecta Predictions - Top):".yellow().bold()
        );

        let trifecta_probs = predictor.calculate_trifecta_probs(&position_probs);

        let has_trifecta_odds = trifecta_odds_map.is_some();
        if has_trifecta_odds {
            println!(
                "{:>10} {:>10} {:>8} {:>8} {:>10}",
                "組合せ", "確率", "オッズ", "EV", "推奨額"
            );
        } else {
            println!("{:>10} {:>10}", "組合せ", "確率");
            println!("{}", "(3連単オッズデータがありません)".dimmed());
        }
        println!("{}", "-".repeat(55));

        let mut trifecta_with_ev: Vec<(TrifectaPrediction, Option<f64>, f64, i64)> = trifecta_probs
            .iter()
            .map(|pred| {
                let odds = trifecta_odds_map
                    .as_ref()
                    .and_then(|o| o.get(&(pred.first, pred.second, pred.third)).copied());
                let ev = odds.map(|o| pred.probability * o).unwrap_or(0.0);
                let stake = if let Some(o) = odds {
                    kelly.calculate_single(pred.probability, o).stake
                } else {
                    0
                };
                (pred.clone(), odds, ev, stake)
            })
            .collect();

        trifecta_with_ev.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        let mut tri_bet_count = 0;
        for (pred, odds, ev, stake) in trifecta_with_ev.iter().take(top) {
            let combo = format!("{}-{}-{}", pred.first, pred.second, pred.third);

            if has_trifecta_odds {
                let odds_str = odds.map(|o| format!("{:.1}", o)).unwrap_or("-".to_string());
                let ev_str = if ev > &0.0 {
                    format!("{:.3}", ev)
                } else {
                    "-".to_string()
                };
                let stake_str = if *stake > 0 {
                    format!("¥{}", stake)
                } else {
                    "-".to_string()
                };

                let ev_color = if *ev >= threshold {
                    tri_bet_count += 1;
                    ev_str.green()
                } else {
                    ev_str.normal()
                };

                println!(
                    "{:>10} {:>9.2}% {:>8} {:>8} {:>10}",
                    combo,
                    pred.probability * 100.0,
                    odds_str,
                    ev_color,
                    stake_str
                );
            } else {
                println!("{:>10} {:>9.2}%", combo, pred.probability * 100.0);
            }
        }
        println!();

        if has_trifecta_odds && tri_bet_count > 0 {
            println!(
                "{} EV ≥ {:.1} のベット候補: {}件",
                "→".green(),
                threshold,
                tri_bet_count
            );
        }
    }

    Ok(())
}

fn list_races(data_dir: &Path, date: u32) -> Result<()> {
    println!(
        "{}: {} ({})",
        "Listing races for".green(),
        date,
        format_date(date)
    );
    println!();

    let csv_path = data_dir.join("programs_entries.csv");

    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg}")
            .unwrap(),
    );
    pb.set_message("Loading race data...");

    let race_data = RaceData::load(&csv_path)
        .with_context(|| format!("Failed to load CSV from {:?}", csv_path))?;

    let races = race_data
        .list_races(date)
        .with_context(|| format!("Failed to list races for {}", date))?;

    pb.finish_and_clear();

    if races.is_empty() {
        println!("{}", "No races found for this date.".yellow());
        return Ok(());
    }

    // Group by stadium
    let mut by_stadium: std::collections::HashMap<u8, Vec<(u8, usize)>> =
        std::collections::HashMap::new();
    for (stadium, race_no, count) in races {
        by_stadium
            .entry(stadium)
            .or_default()
            .push((race_no, count));
    }

    let mut stadiums: Vec<_> = by_stadium.keys().copied().collect();
    stadiums.sort();

    println!(
        "{:>4} {:<10} {:>6} {:>20}",
        "場番", "場名", "レース", "出走数"
    );
    println!("{}", "-".repeat(50));

    for stadium in stadiums {
        let races = by_stadium.get(&stadium).unwrap();
        let total_entries: usize = races.iter().map(|(_, c)| c).sum();

        println!(
            "{:>4} {:<10} {:>6} {:>20}",
            stadium,
            stadium_name(stadium),
            format!("1-{}R", races.len()),
            format!("({} entries)", total_entries)
        );
    }

    println!();
    println!(
        "Total: {} stadiums, {} races",
        by_stadium.len(),
        by_stadium.values().map(|v| v.len()).sum::<usize>()
    );

    Ok(())
}

fn run_backtest(threshold: f64, synthetic_odds: bool) -> Result<()> {
    println!("{}", "Running backtest...".green());
    println!("EV Threshold: {}", threshold);
    println!(
        "Synthetic odds: {}",
        if synthetic_odds { "Yes" } else { "No" }
    );
    println!();

    // TODO: Implement backtesting (Phase 5)
    println!(
        "{}",
        "Note: Backtesting requires implementation (Phase 5)".dimmed()
    );

    Ok(())
}

fn run_interactive(data_dir: &Path, odds_dir: &Path) -> Result<()> {
    println!("{}", "Interactive mode".green().bold());
    println!("Type 'quit' to exit.\n");

    let theme = ColorfulTheme::default();

    loop {
        let options = vec!["Predict a race", "List races", "Backtest", "Quit"];

        let selection = Select::with_theme(&theme)
            .with_prompt("What would you like to do?")
            .items(&options)
            .default(0)
            .interact()?;

        match selection {
            0 => {
                // Predict
                let date: u32 = Input::with_theme(&theme)
                    .with_prompt("Date (YYYYMMDD)")
                    .interact_text()?;

                let stadium: u8 = Input::with_theme(&theme)
                    .with_prompt("Stadium code (1-24)")
                    .interact_text()?;

                let race: u8 = Input::with_theme(&theme)
                    .with_prompt("Race number (1-12)")
                    .interact_text()?;

                let trifecta = Select::with_theme(&theme)
                    .with_prompt("Include trifecta?")
                    .items(&["No", "Yes"])
                    .default(0)
                    .interact()?
                    == 1;

                println!();
                predict_race(
                    data_dir, odds_dir, date, stadium, race, trifecta, 100_000, 0.25, 1.0, 10,
                )?;
                println!();
            }
            1 => {
                // List races
                let date: u32 = Input::with_theme(&theme)
                    .with_prompt("Date (YYYYMMDD)")
                    .interact_text()?;

                println!();
                list_races(data_dir, date)?;
                println!();
            }
            2 => {
                // Backtest
                let threshold: f64 = Input::with_theme(&theme)
                    .with_prompt("EV threshold")
                    .default(1.0)
                    .interact_text()?;

                println!();
                run_backtest(threshold, false)?;
                println!();
            }
            3 => {
                println!("Goodbye!");
                break;
            }
            _ => {}
        }
    }

    Ok(())
}

/// Format date as YYYY-MM-DD
fn format_date(date: u32) -> String {
    let year = date / 10000;
    let month = (date / 100) % 100;
    let day = date % 100;
    format!("{}-{:02}-{:02}", year, month, day)
}

/// Truncate name to fit display width
fn truncate_name(name: &str, max_len: usize) -> String {
    let chars: Vec<char> = name.chars().collect();
    if chars.len() <= max_len {
        name.to_string()
    } else {
        chars[..max_len - 1].iter().collect::<String>() + "…"
    }
}
