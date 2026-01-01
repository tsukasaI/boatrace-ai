//! Race entry (racelist) scraper from boatrace.jp
//!
//! Scrapes race entries (racer information) from the racelist page.

use super::ScraperError;
use crate::models::RacerEntry;
use scraper::{Html, Selector};
use serde::{Deserialize, Serialize};

/// URL for race list page
const RACELIST_URL: &str = "https://www.boatrace.jp/owpc/pc/race/racelist";

/// Scraped race information including entries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScrapedRaceInfo {
    pub date: u32,
    pub stadium_code: u8,
    pub race_no: u8,
    pub race_name: Option<String>,
    pub start_time: Option<String>,
    pub entries: Vec<RacerEntry>,
}

impl ScrapedRaceInfo {
    /// Build URL for racelist page
    pub fn url(date: u32, stadium_code: u8, race_no: u8) -> String {
        format!(
            "{}?rno={}&jcd={:02}&hd={}",
            RACELIST_URL, race_no, stadium_code, date
        )
    }
}

/// Parse race entries from racelist HTML
///
/// The page contains multiple tbody elements (one per boat) with racer information.
/// Structure: tbody.is-fs12 > tr > td elements with racer data
pub fn parse_race_entries(
    html: &str,
    date: u32,
    stadium_code: u8,
    race_no: u8,
) -> Result<ScrapedRaceInfo, ScraperError> {
    let document = Html::parse_document(html);
    let mut entries = Vec::with_capacity(6);

    // Extract race name and start time from header
    let race_name = extract_race_name(&document);
    let start_time = extract_start_time(&document);

    // Each boat is in a separate tbody with class is-fs12
    // Look for tbody elements and find boat numbers via is-boatColor classes
    let tbody_selector = Selector::parse("tbody.is-fs12")
        .map_err(|e| ScraperError::ParseError(e.to_string()))?;

    let boat_cell_selector = Selector::parse("td[class*='is-boatColor'][class*='is-fs14']")
        .map_err(|e| ScraperError::ParseError(e.to_string()))?;

    let racer_link_selector = Selector::parse("a[href*='toban=']")
        .map_err(|e| ScraperError::ParseError(e.to_string()))?;

    for tbody in document.select(&tbody_selector) {
        // Find boat number from the first cell with is-boatColorN is-fs14
        let boat_no = tbody
            .select(&boat_cell_selector)
            .next()
            .and_then(|cell| get_boat_number(&cell));

        let boat_no = match boat_no {
            Some(n) => n,
            None => continue,
        };

        // Find racer ID from link
        let racer_id = tbody
            .select(&racer_link_selector)
            .next()
            .and_then(|link| link.value().attr("href"))
            .and_then(extract_racer_id_from_url);

        let racer_id = match racer_id {
            Some(id) => id,
            None => continue,
        };

        // Find racer name from the link text
        let racer_name_selector = Selector::parse("div.is-fs18 a, a[href*='toban=']").ok();
        let racer_name = racer_name_selector
            .as_ref()
            .and_then(|sel| tbody.select(sel).nth(1)) // Second link usually has the name
            .map(|el| el.text().collect::<String>().trim().replace("　", " "))
            .unwrap_or_default();

        if racer_name.is_empty() {
            continue;
        }

        // Extract other data from the tbody text
        let all_text: String = tbody.text().collect();
        let numbers = extract_numbers(&all_text);

        // Default values
        let mut age: u8 = 30;
        let mut weight: u8 = 52;
        let mut national_win_rate: f64 = 3.0;
        let mut national_in2_rate: f64 = 15.0;
        let mut local_win_rate: f64 = 3.0;
        let mut local_in2_rate: f64 = 15.0;
        let motor_no: u32 = 0;
        let mut motor_in2_rate: f64 = 30.0;
        let boat_no_equip: u32 = 0;
        let mut boat_in2_rate: f64 = 30.0;

        // Try to extract racer class
        let racer_class = extract_racer_class(&tbody);

        // Parse numbers - typical order in HTML:
        // age, weight, F数, L数, ST, national rates, local rates, motor no/rates, boat no/rates
        if numbers.len() >= 10 {
            // Find age and weight (typically small integers together)
            for window in numbers.windows(2) {
                let a = window[0] as u8;
                let w = window[1] as u8;
                if (20..=65).contains(&a) && (45..=60).contains(&w) {
                    age = a;
                    weight = w;
                    break;
                }
            }

            // Find rate pairs (values between 0 and 100)
            let mut rate_pairs: Vec<(f64, f64)> = Vec::new();
            for window in numbers.windows(2) {
                let r1 = window[0];
                let r2 = window[1];
                if (0.0..=10.0).contains(&r1) && (0.0..=100.0).contains(&r2) {
                    rate_pairs.push((r1, r2));
                }
            }

            if rate_pairs.len() >= 4 {
                national_win_rate = rate_pairs[0].0;
                national_in2_rate = rate_pairs[0].1;
                local_win_rate = rate_pairs[1].0;
                local_in2_rate = rate_pairs[1].1;
                motor_in2_rate = rate_pairs[2].1;
                boat_in2_rate = rate_pairs[3].1;
            }
        }

        // Avoid duplicates
        if entries.iter().any(|e: &RacerEntry| e.boat_no == boat_no) {
            continue;
        }

        entries.push(RacerEntry {
            boat_no,
            racer_id,
            racer_name,
            age,
            weight,
            racer_class: racer_class.unwrap_or_else(|| "B1".to_string()),
            branch: String::new(), // Branch not available from web scraping
            national_win_rate,
            national_in2_rate,
            local_win_rate,
            local_in2_rate,
            motor_no,
            motor_in2_rate,
            boat_no_equip,
            boat_in2_rate,
        });
    }

    // Sort by boat number
    entries.sort_by_key(|e| e.boat_no);

    if entries.is_empty() {
        return Err(ScraperError::ParseError(
            "No racer entries found in HTML".to_string(),
        ));
    }

    if entries.len() != 6 {
        tracing::warn!(
            "Expected 6 entries, got {} for race {}-{}-{}",
            entries.len(),
            date,
            stadium_code,
            race_no
        );
    }

    Ok(ScrapedRaceInfo {
        date,
        stadium_code,
        race_no,
        race_name,
        start_time,
        entries,
    })
}

/// Extract racer class from tbody
fn extract_racer_class(tbody: &scraper::ElementRef) -> Option<String> {
    let class_selector = Selector::parse("span").ok()?;
    for span in tbody.select(&class_selector) {
        let text: String = span.text().collect::<String>().trim().to_string();
        if is_racer_class(&text) {
            return Some(text);
        }
    }
    None
}

/// Get boat number from cell (by class or text)
fn get_boat_number(cell: &scraper::ElementRef) -> Option<u8> {
    // Check CSS class
    for class in cell.value().classes() {
        for i in 1..=6 {
            if class.contains(&format!("is-boatColor{}", i)) {
                return Some(i);
            }
        }
    }

    // Check text
    let text: String = cell.text().collect::<String>().trim().to_string();
    if let Ok(num) = text.parse::<u8>() {
        if (1..=6).contains(&num) {
            return Some(num);
        }
    }

    None
}

/// Extract racer ID from URL (toban parameter)
fn extract_racer_id_from_url(url: &str) -> Option<u32> {
    if let Some(idx) = url.find("toban=") {
        let rest = &url[idx + 6..];
        let id_str: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
        id_str.parse::<u32>().ok()
    } else {
        None
    }
}

/// Check if text is a racer class
fn is_racer_class(text: &str) -> bool {
    matches!(text.trim(), "A1" | "A2" | "B1" | "B2")
}

/// Extract all numbers from text
fn extract_numbers(text: &str) -> Vec<f64> {
    let mut numbers = Vec::new();
    let mut current = String::new();
    let mut has_dot = false;

    for ch in text.chars() {
        if ch.is_ascii_digit() {
            current.push(ch);
        } else if ch == '.' && !has_dot && !current.is_empty() {
            current.push(ch);
            has_dot = true;
        } else if !current.is_empty() {
            if let Ok(num) = current.parse::<f64>() {
                numbers.push(num);
            }
            current.clear();
            has_dot = false;
        }
    }

    if !current.is_empty() {
        if let Ok(num) = current.parse::<f64>() {
            numbers.push(num);
        }
    }

    numbers
}

/// Extract race name from document
fn extract_race_name(document: &Html) -> Option<String> {
    let selector = Selector::parse("h2.heading2_titleName, span.heading2_titleName").ok()?;

    document.select(&selector).next().map(|el| {
        el.text()
            .collect::<String>()
            .trim()
            .to_string()
    })
}

/// Extract start time from document
fn extract_start_time(document: &Html) -> Option<String> {
    let selector = Selector::parse("span.heading2_titleDetail, div.heading2_titleDetail").ok()?;

    for el in document.select(&selector) {
        let text: String = el.text().collect::<String>().trim().to_string();
        // Look for time pattern (e.g., "14:30" or "締切予定 14:30")
        if text.contains(':') {
            let parts: Vec<&str> = text.split_whitespace().collect();
            for part in parts {
                if part.contains(':') && part.len() <= 5 {
                    return Some(part.to_string());
                }
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_racer_info_url() {
        let url = ScrapedRaceInfo::url(20251231, 23, 8);
        assert_eq!(
            url,
            "https://www.boatrace.jp/owpc/pc/race/racelist?rno=8&jcd=23&hd=20251231"
        );
    }

    #[test]
    fn test_extract_racer_id_from_url() {
        assert_eq!(
            extract_racer_id_from_url("?toban=4444&hd=20251231"),
            Some(4444)
        );
        assert_eq!(
            extract_racer_id_from_url("/racer/profile?toban=12345"),
            Some(12345)
        );
        assert_eq!(extract_racer_id_from_url("no-id"), None);
    }

    #[test]
    fn test_is_racer_class() {
        assert!(is_racer_class("A1"));
        assert!(is_racer_class("B2"));
        assert!(!is_racer_class("A3"));
        assert!(!is_racer_class(""));
    }

    #[test]
    fn test_extract_numbers() {
        let nums = extract_numbers("Age 35 Weight 52 Rate 5.32");
        assert_eq!(nums, vec![35.0, 52.0, 5.32]);
    }

    #[test]
    fn test_parse_empty_html() {
        let html = "<html><body></body></html>";
        let result = parse_race_entries(html, 20251231, 23, 1);
        assert!(result.is_err());
    }
}
