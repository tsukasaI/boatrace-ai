//! Web scraper for boatrace.jp
//!
//! Scrapes odds, race schedules, and race entries from the official website.
//!
//! # Example
//!
//! ```no_run
//! use boatrace::scraper::{OddsScraper, ScraperConfig};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let scraper = OddsScraper::new(ScraperConfig::default());
//!
//!     // Scrape single race exacta odds
//!     let odds = scraper.scrape_exacta(20241230, 23, 1).await?;
//!     println!("Found {} combinations", odds.exacta.len());
//!
//!     // Scrape today's schedule
//!     let schedule = scraper.scrape_schedule(20251231).await?;
//!     println!("Found {} active stadiums", schedule.stadiums.len());
//!
//!     // Scrape race entries
//!     let race = scraper.scrape_race_entries(20251231, 23, 1).await?;
//!     println!("Found {} entries", race.entries.len());
//!
//!     Ok(())
//! }
//! ```

mod client;
mod entries;
mod exacta;
mod schedule;
mod trifecta;

pub use client::{OddsScraper, ScraperConfig, ScraperError};
pub use entries::{parse_race_deadlines, parse_race_entries, ScrapedRaceInfo};
pub use exacta::parse_exacta_odds;
pub use schedule::{parse_schedule, ActiveStadium, TodaySchedule};
pub use trifecta::parse_trifecta_odds;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Exacta odds data for a single race
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScrapedExactaOdds {
    pub date: u32,
    pub stadium_code: u8,
    pub race_no: u8,
    pub scraped_at: String,
    /// Odds map: key = "first-second" (e.g., "1-2"), value = odds
    pub exacta: HashMap<String, f64>,
}

impl ScrapedExactaOdds {
    /// Get odds for a specific combination
    pub fn get(&self, first: u8, second: u8) -> Option<f64> {
        let key = format!("{}-{}", first, second);
        self.exacta.get(&key).copied()
    }

    /// Convert to (first, second) -> odds map
    pub fn to_tuple_map(&self) -> HashMap<(u8, u8), f64> {
        self.exacta
            .iter()
            .filter_map(|(k, &v)| {
                let parts: Vec<&str> = k.split('-').collect();
                if parts.len() == 2 {
                    let first = parts[0].parse::<u8>().ok()?;
                    let second = parts[1].parse::<u8>().ok()?;
                    Some(((first, second), v))
                } else {
                    None
                }
            })
            .collect()
    }
}

/// Trifecta odds data for a single race
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScrapedTrifectaOdds {
    pub date: u32,
    pub stadium_code: u8,
    pub race_no: u8,
    pub scraped_at: String,
    /// Odds map: key = "first-second-third" (e.g., "1-2-3"), value = odds
    pub trifecta: HashMap<String, f64>,
}

impl ScrapedTrifectaOdds {
    /// Get odds for a specific combination
    pub fn get(&self, first: u8, second: u8, third: u8) -> Option<f64> {
        let key = format!("{}-{}-{}", first, second, third);
        self.trifecta.get(&key).copied()
    }

    /// Convert to (first, second, third) -> odds map
    pub fn to_tuple_map(&self) -> HashMap<(u8, u8, u8), f64> {
        self.trifecta
            .iter()
            .filter_map(|(k, &v)| {
                let parts: Vec<&str> = k.split('-').collect();
                if parts.len() == 3 {
                    let first = parts[0].parse::<u8>().ok()?;
                    let second = parts[1].parse::<u8>().ok()?;
                    let third = parts[2].parse::<u8>().ok()?;
                    Some(((first, second, third), v))
                } else {
                    None
                }
            })
            .collect()
    }
}

/// A timestamped pre-deadline odds snapshot for lookahead-free backtesting.
///
/// Unlike [`ScrapedExactaOdds`]/[`ScrapedTrifectaOdds`] (which overwrite a single
/// file per race), snapshots are never overwritten: each is saved under a
/// filename embedding the capture time, so multiple snapshots per race coexist.
/// `deadline` records the race's `締切予定時刻` so a later backtest can keep only
/// snapshots captured before betting closed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OddsSnapshot {
    pub date: u32,
    pub stadium_code: u8,
    pub race_no: u8,
    /// RFC3339 capture time (from the underlying scrape).
    pub scraped_at: String,
    /// Race deadline `HH:MM` (`締切予定時刻`), if known.
    pub deadline: Option<String>,
    /// Whether `deadline` is the scraped `締切予定時刻` (`true`) or was
    /// approximated from `race_no` via [`approximate_deadline`] (`false`).
    /// `#[serde(default)]` keeps snapshots written before this field existed
    /// loadable; they default to `false`, the conservative (fail-closed) choice.
    #[serde(default)]
    pub deadline_exact: bool,
    /// `"exacta"` or `"trifecta"`.
    pub bet_type: String,
    /// Odds map: key `"1-2"` (exacta) or `"1-2-3"` (trifecta) -> odds.
    pub odds: HashMap<String, f64>,
}

/// Approximate a race's betting deadline (`HH:MM`, JST) from its number, for use
/// only when the scraped `締切予定時刻` is unavailable. Boatrace days start around
/// 10:30 with races roughly 35 minutes apart. Snapshots built from this estimate
/// must set `deadline_exact = false` so a lookahead-free backtest can treat them
/// conservatively (a wrong estimate could otherwise schedule a post-close capture).
#[must_use]
pub fn approximate_deadline(race_no: u8) -> String {
    let total = 10 * 60 + 30 + u16::from(race_no.saturating_sub(1)) * 35;
    format!("{:02}:{:02}", total / 60, total % 60)
}

impl OddsSnapshot {
    /// Filename embedding the capture timestamp so snapshots never overwrite each
    /// other, e.g. `20260620_23_01_20260620T142233...Z.json` (trifecta gets a
    /// `_3t` suffix). The RFC3339 timestamp is reduced to its alphanumeric
    /// characters to stay filesystem-safe while remaining lexically sortable.
    #[must_use]
    pub fn filename(&self) -> String {
        let ts: String = self
            .scraped_at
            .chars()
            .filter(char::is_ascii_alphanumeric)
            .collect();
        let suffix = if self.bet_type == "trifecta" {
            "_3t"
        } else {
            ""
        };
        format!(
            "{}_{:02}_{:02}_{}{}.json",
            self.date, self.stadium_code, self.race_no, ts, suffix
        )
    }
}

/// Save a snapshot under `dir`, never overwriting an existing capture, and return
/// the written path. The caller is responsible for creating `dir`.
pub fn save_snapshot(dir: &Path, snapshot: &OddsSnapshot) -> std::io::Result<PathBuf> {
    let path = dir.join(snapshot.filename());
    let json = serde_json::to_string_pretty(snapshot).map_err(std::io::Error::other)?;
    std::fs::write(&path, json)?;
    Ok(path)
}

/// Stadium codes and names
pub fn get_stadium_name(code: u8) -> &'static str {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exacta_odds_get() {
        let mut exacta = HashMap::new();
        exacta.insert("1-2".to_string(), 5.5);
        exacta.insert("2-1".to_string(), 8.0);

        let odds = ScrapedExactaOdds {
            date: 20241230,
            stadium_code: 23,
            race_no: 1,
            scraped_at: "2024-12-30T10:00:00".to_string(),
            exacta,
        };

        assert_eq!(odds.get(1, 2), Some(5.5));
        assert_eq!(odds.get(2, 1), Some(8.0));
        assert_eq!(odds.get(1, 3), None);
    }

    #[test]
    fn test_exacta_odds_to_tuple_map() {
        let mut exacta = HashMap::new();
        exacta.insert("1-2".to_string(), 5.5);
        exacta.insert("3-4".to_string(), 10.0);

        let odds = ScrapedExactaOdds {
            date: 20241230,
            stadium_code: 23,
            race_no: 1,
            scraped_at: "2024-12-30T10:00:00".to_string(),
            exacta,
        };

        let tuple_map = odds.to_tuple_map();
        assert_eq!(tuple_map.get(&(1, 2)), Some(&5.5));
        assert_eq!(tuple_map.get(&(3, 4)), Some(&10.0));
    }

    #[test]
    fn test_trifecta_odds_get() {
        let mut trifecta = HashMap::new();
        trifecta.insert("1-2-3".to_string(), 25.5);

        let odds = ScrapedTrifectaOdds {
            date: 20241230,
            stadium_code: 23,
            race_no: 1,
            scraped_at: "2024-12-30T10:00:00".to_string(),
            trifecta,
        };

        assert_eq!(odds.get(1, 2, 3), Some(25.5));
        assert_eq!(odds.get(1, 2, 4), None);
    }

    #[test]
    fn test_stadium_names() {
        assert_eq!(get_stadium_name(23), "唐津");
        assert_eq!(get_stadium_name(1), "桐生");
        assert_eq!(get_stadium_name(24), "大村");
        assert_eq!(get_stadium_name(99), "不明");
    }

    fn sample_snapshot(scraped_at: &str, bet_type: &str) -> OddsSnapshot {
        let mut odds = HashMap::new();
        odds.insert("1-2".to_string(), 5.6);
        OddsSnapshot {
            date: 20260620,
            stadium_code: 23,
            race_no: 1,
            scraped_at: scraped_at.to_string(),
            deadline: Some("15:24".to_string()),
            deadline_exact: true,
            bet_type: bet_type.to_string(),
            odds,
        }
    }

    #[test]
    fn test_approximate_deadline() {
        assert_eq!(approximate_deadline(1), "10:30");
        assert_eq!(approximate_deadline(2), "11:05");
        assert_eq!(approximate_deadline(12), "16:55");
        // race_no 0 is nonsensical but must not panic; saturating_sub keeps it at R1.
        assert_eq!(approximate_deadline(0), "10:30");
    }

    #[test]
    fn test_snapshot_deadline_exact_defaults_false_for_legacy_json() {
        // A snapshot written before `deadline_exact` existed must still load,
        // defaulting to false (fail-closed: treated as approximate/unknown).
        let legacy = r#"{
            "date": 20260620,
            "stadium_code": 23,
            "race_no": 1,
            "scraped_at": "2026-06-20T14:22:33+09:00",
            "deadline": "15:24",
            "bet_type": "exacta",
            "odds": { "1-2": 5.6 }
        }"#;
        let snapshot: OddsSnapshot = serde_json::from_str(legacy).unwrap();
        assert!(!snapshot.deadline_exact);
        assert_eq!(snapshot.deadline.as_deref(), Some("15:24"));
    }

    #[test]
    fn test_snapshot_filename_is_safe_and_typed() {
        let exacta = sample_snapshot("2026-06-20T14:22:33+09:00", "exacta");
        let name = exacta.filename();
        // Filesystem-safe: no ':' or '+' from the RFC3339 timestamp survive.
        assert!(!name.contains(':') && !name.contains('+'));
        assert!(name.starts_with("20260620_23_01_"));
        assert!(name.ends_with(".json") && !name.contains("_3t"));

        // Trifecta is disambiguated by the _3t suffix.
        let tri = sample_snapshot("2026-06-20T14:22:33+09:00", "trifecta");
        assert!(tri.filename().ends_with("_3t.json"));
    }

    #[test]
    fn test_save_snapshot_never_overwrites() {
        // Two captures of the same race at different times must coexist.
        let dir = std::env::temp_dir().join("boatrace_snapshot_test_no_overwrite");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let first = sample_snapshot("2026-06-20T14:22:33+09:00", "exacta");
        let second = sample_snapshot("2026-06-20T14:25:01+09:00", "exacta");
        assert_ne!(first.filename(), second.filename());

        let p1 = save_snapshot(&dir, &first).unwrap();
        let p2 = save_snapshot(&dir, &second).unwrap();
        assert!(p1.exists() && p2.exists());
        assert_ne!(p1, p2);

        let count = std::fs::read_dir(&dir).unwrap().count();
        assert_eq!(count, 2, "both snapshots should survive");

        // Round-trips back to an equivalent struct.
        let reloaded: OddsSnapshot =
            serde_json::from_str(&std::fs::read_to_string(&p1).unwrap()).unwrap();
        assert_eq!(reloaded.deadline.as_deref(), Some("15:24"));
        assert_eq!(reloaded.odds.get("1-2"), Some(&5.6));

        std::fs::remove_dir_all(&dir).unwrap();
    }
}
