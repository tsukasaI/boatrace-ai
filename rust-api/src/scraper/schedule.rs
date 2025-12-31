//! Today's race schedule scraper from boatrace.jp
//!
//! Scrapes the race index page to detect which stadiums have active races today.

use super::ScraperError;
use scraper::{Html, Selector};
use serde::{Deserialize, Serialize};

/// URL for race index page
const RACE_INDEX_URL: &str = "https://www.boatrace.jp/owpc/pc/race/index";

/// Today's race schedule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TodaySchedule {
    pub date: u32,
    pub scraped_at: String,
    pub stadiums: Vec<ActiveStadium>,
}

/// Active stadium information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveStadium {
    pub code: u8,
    pub name: String,
    pub is_selling: bool,
    pub current_race: Option<u8>,
    pub total_races: u8,
    pub event_name: Option<String>,
    pub grade: Option<String>,
}

impl TodaySchedule {
    /// Get URL for race index page
    pub fn url() -> &'static str {
        RACE_INDEX_URL
    }
}

/// Parse today's schedule from race index HTML
///
/// The page shows all active stadiums with their current race status.
/// Stadium links contain jcd parameter with stadium code.
pub fn parse_schedule(html: &str, date: u32) -> Result<TodaySchedule, ScraperError> {
    let document = Html::parse_document(html);
    let mut stadiums = Vec::new();

    // Look for stadium table - each row contains stadium info
    // The table has class "table1" with stadium links
    let table_selector = Selector::parse("div.table1 table tbody tr")
        .map_err(|e| ScraperError::ParseError(e.to_string()))?;

    // Alternative: look for stadium cards/links
    let stadium_link_selector = Selector::parse("a[href*='jcd=']")
        .map_err(|e| ScraperError::ParseError(e.to_string()))?;

    // Try table-based parsing first
    for row in document.select(&table_selector) {
        if let Some(stadium) = parse_stadium_row(&row) {
            stadiums.push(stadium);
        }
    }

    // If no stadiums found from table, try link-based parsing
    if stadiums.is_empty() {
        let mut seen_codes: std::collections::HashSet<u8> = std::collections::HashSet::new();

        for link in document.select(&stadium_link_selector) {
            if let Some(href) = link.value().attr("href") {
                if let Some(code) = extract_stadium_code(href) {
                    if !seen_codes.contains(&code) {
                        seen_codes.insert(code);

                        let name = super::get_stadium_name(code).to_string();
                        let event_name = extract_event_name(&link);
                        let grade = extract_grade(&link);

                        stadiums.push(ActiveStadium {
                            code,
                            name,
                            is_selling: true, // Assume selling if listed
                            current_race: None,
                            total_races: 12, // Default
                            event_name,
                            grade,
                        });
                    }
                }
            }
        }
    }

    // Sort by stadium code
    stadiums.sort_by_key(|s| s.code);

    Ok(TodaySchedule {
        date,
        scraped_at: chrono::Utc::now().to_rfc3339(),
        stadiums,
    })
}

/// Parse a stadium row from the table
fn parse_stadium_row(row: &scraper::ElementRef) -> Option<ActiveStadium> {
    let td_selector = Selector::parse("td").ok()?;
    let link_selector = Selector::parse("a[href*='jcd=']").ok()?;

    // Find stadium link in the row
    let link = row.select(&link_selector).next()?;
    let href = link.value().attr("href")?;
    let code = extract_stadium_code(href)?;

    let name = super::get_stadium_name(code).to_string();

    // Look for race status (current race number, selling status)
    let mut is_selling = true;
    let mut current_race: Option<u8> = None;
    let mut grade: Option<String> = None;

    for td in row.select(&td_selector) {
        let text: String = td.text().collect::<String>().trim().to_string();

        // Check for selling status
        if text.contains("発売中") {
            is_selling = true;
        } else if text.contains("終了") || text.contains("中止") {
            is_selling = false;
        }

        // Look for race number (e.g., "8R", "R8")
        if let Some(race_num) = extract_race_number(&text) {
            current_race = Some(race_num);
        }

        // Look for event/grade info
        if text.contains("SG") || text.contains("G1") || text.contains("G2") || text.contains("G3") {
            grade = Some(text.clone());
        }
    }

    // Try to get event name from link or nearby elements
    let event_name = extract_event_name(&link);
    if grade.is_none() {
        grade = extract_grade(&link);
    }

    Some(ActiveStadium {
        code,
        name,
        is_selling,
        current_race,
        total_races: 12,
        event_name,
        grade,
    })
}

/// Extract stadium code from URL (jcd parameter)
fn extract_stadium_code(url: &str) -> Option<u8> {
    // Look for jcd=XX pattern
    if let Some(idx) = url.find("jcd=") {
        let rest = &url[idx + 4..];
        let code_str: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
        code_str.parse::<u8>().ok()
    } else {
        None
    }
}

/// Extract race number from text (e.g., "8R", "R8", "第8R")
fn extract_race_number(text: &str) -> Option<u8> {
    // Pattern: "8R" or "R8" or "第8R"
    let text = text.replace("第", "");

    if text.contains('R') {
        let parts: Vec<&str> = text.split('R').collect();
        if parts.len() >= 2 {
            // Try "8R" format (number before R)
            if let Ok(num) = parts[0].trim().parse::<u8>() {
                if (1..=12).contains(&num) {
                    return Some(num);
                }
            }
            // Try "R8" format (number after R)
            if let Ok(num) = parts[1].trim().chars().take_while(|c| c.is_ascii_digit()).collect::<String>().parse::<u8>() {
                if (1..=12).contains(&num) {
                    return Some(num);
                }
            }
        }
    }
    None
}

/// Extract event name from link element
fn extract_event_name(link: &scraper::ElementRef) -> Option<String> {
    // Look for title attribute or text content
    if let Some(title) = link.value().attr("title") {
        if !title.is_empty() {
            return Some(title.to_string());
        }
    }

    // Look for event name in nested span/text
    let text: String = link.text().collect::<String>().trim().to_string();
    if !text.is_empty() && text.len() > 2 {
        Some(text)
    } else {
        None
    }
}

/// Extract grade from link element or parent
fn extract_grade(link: &scraper::ElementRef) -> Option<String> {
    let classes: Vec<&str> = link.value().classes().collect();

    for class in &classes {
        if class.contains("is-SG") || class.contains("SG") {
            return Some("SG".to_string());
        } else if class.contains("is-G1") || class.contains("G1") {
            return Some("G1".to_string());
        } else if class.contains("is-G2") || class.contains("G2") {
            return Some("G2".to_string());
        } else if class.contains("is-G3") || class.contains("G3") {
            return Some("G3".to_string());
        }
    }

    // Check text content
    let text: String = link.text().collect();
    if text.contains("SG") {
        Some("SG".to_string())
    } else if text.contains("G1") {
        Some("G1".to_string())
    } else if text.contains("G2") {
        Some("G2".to_string())
    } else if text.contains("G3") {
        Some("G3".to_string())
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_stadium_code() {
        assert_eq!(extract_stadium_code("?jcd=23&hd=20251231"), Some(23));
        assert_eq!(extract_stadium_code("race/odds?jcd=01&rno=1"), Some(1));
        assert_eq!(extract_stadium_code("?jcd=24"), Some(24));
        assert_eq!(extract_stadium_code("no-code"), None);
    }

    #[test]
    fn test_extract_race_number() {
        assert_eq!(extract_race_number("8R"), Some(8));
        assert_eq!(extract_race_number("R8"), Some(8));
        assert_eq!(extract_race_number("第8R"), Some(8));
        assert_eq!(extract_race_number("12R発売中"), Some(12));
        assert_eq!(extract_race_number("no race"), None);
    }

    #[test]
    fn test_parse_empty_html() {
        let html = "<html><body></body></html>";
        let result = parse_schedule(html, 20251231);
        assert!(result.is_ok());
        assert!(result.unwrap().stadiums.is_empty());
    }

    #[test]
    fn test_parse_with_stadium_links() {
        let html = r#"
        <html><body>
            <a href="/race/index?jcd=23&hd=20251231">唐津</a>
            <a href="/race/index?jcd=12&hd=20251231">住之江</a>
        </body></html>
        "#;
        let result = parse_schedule(html, 20251231).unwrap();
        assert_eq!(result.stadiums.len(), 2);
        assert_eq!(result.stadiums[0].code, 12); // Sorted
        assert_eq!(result.stadiums[1].code, 23);
    }
}
