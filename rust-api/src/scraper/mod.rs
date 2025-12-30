//! Web scraper for boatrace.jp odds
//!
//! Scrapes exacta (2連単) and trifecta (3連単) odds from the official website.
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
//!     Ok(())
//! }
//! ```

mod client;
mod exacta;
mod trifecta;

pub use client::{OddsScraper, ScraperConfig, ScraperError};
pub use exacta::parse_exacta_odds;
pub use trifecta::parse_trifecta_odds;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
}
