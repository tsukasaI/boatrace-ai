//! Odds JSON loading for exacta and trifecta odds

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Exacta odds JSON structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExactaOddsFile {
    pub date: u32,
    pub stadium_code: u8,
    pub race_no: u8,
    pub scraped_at: String,
    pub exacta: HashMap<String, f64>,
}

/// Trifecta odds JSON structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrifectaOddsFile {
    pub date: u32,
    pub stadium_code: u8,
    pub race_no: u8,
    pub scraped_at: String,
    pub trifecta: HashMap<String, f64>,
}

/// Load exacta odds from JSON file
///
/// Returns a HashMap with (first, second) -> odds mapping
pub fn load_exacta_odds<P: AsRef<Path>>(
    odds_dir: P,
    date: u32,
    stadium_code: u8,
    race_no: u8,
) -> Option<HashMap<(u8, u8), f64>> {
    let filename = format!("{}_{:02}_{:02}.json", date, stadium_code, race_no);
    let path = odds_dir.as_ref().join(&filename);

    let content = fs::read_to_string(&path).ok()?;
    let odds_file: ExactaOddsFile = serde_json::from_str(&content).ok()?;

    let mut result = HashMap::new();
    for (key, odds) in odds_file.exacta {
        if let Some((first, second)) = parse_exacta_key(&key) {
            result.insert((first, second), odds);
        }
    }

    Some(result)
}

/// Load trifecta odds from JSON file
///
/// Returns a HashMap with (first, second, third) -> odds mapping
pub fn load_trifecta_odds<P: AsRef<Path>>(
    odds_dir: P,
    date: u32,
    stadium_code: u8,
    race_no: u8,
) -> Option<HashMap<(u8, u8, u8), f64>> {
    let filename = format!("{}_{:02}_{:02}_3t.json", date, stadium_code, race_no);
    let path = odds_dir.as_ref().join(&filename);

    let content = fs::read_to_string(&path).ok()?;
    let odds_file: TrifectaOddsFile = serde_json::from_str(&content).ok()?;

    let mut result = HashMap::new();
    for (key, odds) in odds_file.trifecta {
        if let Some((first, second, third)) = parse_trifecta_key(&key) {
            result.insert((first, second, third), odds);
        }
    }

    Some(result)
}

/// Parse exacta key "1-2" to (1, 2)
fn parse_exacta_key(key: &str) -> Option<(u8, u8)> {
    let parts: Vec<&str> = key.split('-').collect();
    if parts.len() != 2 {
        return None;
    }
    let first: u8 = parts[0].parse().ok()?;
    let second: u8 = parts[1].parse().ok()?;
    Some((first, second))
}

/// Parse trifecta key "1-2-3" to (1, 2, 3)
fn parse_trifecta_key(key: &str) -> Option<(u8, u8, u8)> {
    let parts: Vec<&str> = key.split('-').collect();
    if parts.len() != 3 {
        return None;
    }
    let first: u8 = parts[0].parse().ok()?;
    let second: u8 = parts[1].parse().ok()?;
    let third: u8 = parts[2].parse().ok()?;
    Some((first, second, third))
}

/// Check if exacta odds file exists
pub fn exacta_odds_exists<P: AsRef<Path>>(
    odds_dir: P,
    date: u32,
    stadium_code: u8,
    race_no: u8,
) -> bool {
    let filename = format!("{}_{:02}_{:02}.json", date, stadium_code, race_no);
    odds_dir.as_ref().join(&filename).exists()
}

/// Check if trifecta odds file exists
pub fn trifecta_odds_exists<P: AsRef<Path>>(
    odds_dir: P,
    date: u32,
    stadium_code: u8,
    race_no: u8,
) -> bool {
    let filename = format!("{}_{:02}_{:02}_3t.json", date, stadium_code, race_no);
    odds_dir.as_ref().join(&filename).exists()
}

/// List all available odds files in directory
pub fn list_odds_files<P: AsRef<Path>>(odds_dir: P) -> Vec<(u32, u8, u8, bool)> {
    let mut results = Vec::new();

    if let Ok(entries) = fs::read_dir(odds_dir) {
        for entry in entries.flatten() {
            if let Some(filename) = entry.file_name().to_str() {
                // Parse filename: {date}_{stadium:02}_{race:02}.json or _3t.json
                if filename.ends_with("_3t.json") {
                    // Trifecta file
                    let base = filename.trim_end_matches("_3t.json");
                    if let Some((date, stadium, race)) = parse_filename(base) {
                        results.push((date, stadium, race, true));
                    }
                } else if filename.ends_with(".json") && !filename.contains("_3t") {
                    // Exacta file
                    let base = filename.trim_end_matches(".json");
                    if let Some((date, stadium, race)) = parse_filename(base) {
                        results.push((date, stadium, race, false));
                    }
                }
            }
        }
    }

    results.sort();
    results
}

/// Parse filename "20240115_03_01" to (date, stadium, race)
fn parse_filename(base: &str) -> Option<(u32, u8, u8)> {
    let parts: Vec<&str> = base.split('_').collect();
    if parts.len() != 3 {
        return None;
    }
    let date: u32 = parts[0].parse().ok()?;
    let stadium: u8 = parts[1].parse().ok()?;
    let race: u8 = parts[2].parse().ok()?;
    Some((date, stadium, race))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_exacta_key() {
        assert_eq!(parse_exacta_key("1-2"), Some((1, 2)));
        assert_eq!(parse_exacta_key("6-5"), Some((6, 5)));
        assert_eq!(parse_exacta_key("invalid"), None);
        assert_eq!(parse_exacta_key("1-2-3"), None);
    }

    #[test]
    fn test_parse_trifecta_key() {
        assert_eq!(parse_trifecta_key("1-2-3"), Some((1, 2, 3)));
        assert_eq!(parse_trifecta_key("6-5-4"), Some((6, 5, 4)));
        assert_eq!(parse_trifecta_key("invalid"), None);
        assert_eq!(parse_trifecta_key("1-2"), None);
    }

    #[test]
    fn test_parse_filename() {
        assert_eq!(parse_filename("20240115_03_01"), Some((20240115, 3, 1)));
        assert_eq!(parse_filename("20231231_24_12"), Some((20231231, 24, 12)));
        assert_eq!(parse_filename("invalid"), None);
    }

    #[test]
    fn test_load_exacta_from_json_string() {
        let json = r#"{
            "date": 20240115,
            "stadium_code": 3,
            "race_no": 1,
            "scraped_at": "2025-12-30T21:03:54.919741",
            "exacta": {
                "1-2": 7.6,
                "2-1": 20.3
            }
        }"#;

        let odds_file: ExactaOddsFile = serde_json::from_str(json).unwrap();
        assert_eq!(odds_file.date, 20240115);
        assert_eq!(odds_file.stadium_code, 3);
        assert_eq!(odds_file.race_no, 1);
        assert_eq!(odds_file.exacta.len(), 2);
        assert!((odds_file.exacta["1-2"] - 7.6).abs() < 0.01);
    }
}
