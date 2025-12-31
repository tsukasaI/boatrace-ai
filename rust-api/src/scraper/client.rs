//! HTTP client with rate limiting for boatrace.jp

use super::{
    parse_exacta_odds, parse_race_entries, parse_schedule, parse_trifecta_odds,
    ScrapedExactaOdds, ScrapedRaceInfo, ScrapedTrifectaOdds, TodaySchedule,
};
use chrono::Utc;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;
use tokio::sync::Mutex;

/// Base URLs for pages
const BASE_URL_EXACTA: &str = "https://www.boatrace.jp/owpc/pc/race/odds2tf";
const BASE_URL_TRIFECTA: &str = "https://www.boatrace.jp/owpc/pc/race/odds3t";
const BASE_URL_RACELIST: &str = "https://www.boatrace.jp/owpc/pc/race/racelist";
const BASE_URL_INDEX: &str = "https://www.boatrace.jp/owpc/pc/race/index";

/// Scraper errors
#[derive(Debug, Error)]
pub enum ScraperError {
    #[error("HTTP request failed: {0}")]
    RequestFailed(#[from] reqwest::Error),

    #[error("Failed to parse HTML: {0}")]
    ParseError(String),

    #[error("No odds found for race")]
    NoOddsFound,

    #[error("Unexpected combination count: expected {expected}, got {actual}")]
    UnexpectedCombinations { expected: usize, actual: usize },
}

/// Scraper configuration
#[derive(Debug, Clone)]
pub struct ScraperConfig {
    /// Delay between requests in milliseconds
    pub delay_ms: u64,
    /// Request timeout in seconds
    pub timeout_secs: u64,
    /// Max retry attempts
    pub max_retries: u32,
    /// User agent string
    pub user_agent: String,
}

impl Default for ScraperConfig {
    fn default() -> Self {
        Self {
            delay_ms: 2000,
            timeout_secs: 30,
            max_retries: 3,
            user_agent: "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36".to_string(),
        }
    }
}

/// Odds scraper with rate limiting
pub struct OddsScraper {
    client: reqwest::Client,
    config: ScraperConfig,
    last_request: Arc<Mutex<Instant>>,
}

impl OddsScraper {
    /// Create a new scraper with the given configuration
    pub fn new(config: ScraperConfig) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .user_agent(&config.user_agent)
            .build()
            .expect("Failed to create HTTP client");

        Self {
            client,
            config,
            last_request: Arc::new(Mutex::new(Instant::now() - Duration::from_secs(10))),
        }
    }

    /// Wait for rate limit
    async fn wait_for_rate_limit(&self) {
        let mut last = self.last_request.lock().await;
        let elapsed = last.elapsed();
        let delay = Duration::from_millis(self.config.delay_ms);

        if elapsed < delay {
            tokio::time::sleep(delay - elapsed).await;
        }

        *last = Instant::now();
    }

    /// Build URL for odds page
    fn build_url(&self, date: u32, stadium_code: u8, race_no: u8, bet_type: &str) -> String {
        let base = if bet_type == "trifecta" {
            BASE_URL_TRIFECTA
        } else {
            BASE_URL_EXACTA
        };

        format!("{}?rno={}&jcd={:02}&hd={}", base, race_no, stadium_code, date)
    }

    /// Fetch HTML page with rate limiting and retry
    async fn fetch_page(&self, url: &str) -> Result<String, ScraperError> {
        for attempt in 0..self.config.max_retries {
            self.wait_for_rate_limit().await;

            match self.client.get(url).send().await {
                Ok(response) => {
                    if response.status().is_success() {
                        return response.text().await.map_err(ScraperError::RequestFailed);
                    }
                    tracing::warn!(
                        "Request failed with status {} (attempt {}/{})",
                        response.status(),
                        attempt + 1,
                        self.config.max_retries
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        "Request failed (attempt {}/{}): {}",
                        attempt + 1,
                        self.config.max_retries,
                        e
                    );
                }
            }

            if attempt + 1 < self.config.max_retries {
                let backoff = Duration::from_millis(self.config.delay_ms * (attempt as u64 + 1));
                tokio::time::sleep(backoff).await;
            }
        }

        Err(ScraperError::ParseError(format!(
            "Failed to fetch {} after {} attempts",
            url, self.config.max_retries
        )))
    }

    /// Scrape exacta odds for a single race
    pub async fn scrape_exacta(
        &self,
        date: u32,
        stadium_code: u8,
        race_no: u8,
    ) -> Result<ScrapedExactaOdds, ScraperError> {
        let url = self.build_url(date, stadium_code, race_no, "exacta");
        tracing::info!("Scraping exacta: {}", url);

        let html = self.fetch_page(&url).await?;
        let odds = parse_exacta_odds(&html)?;

        if odds.is_empty() {
            return Err(ScraperError::NoOddsFound);
        }

        // Expect 30 combinations (6 * 5)
        if odds.len() != 30 {
            tracing::warn!("Expected 30 combinations, got {}", odds.len());
        }

        // Convert to string keys
        let exacta: HashMap<String, f64> = odds
            .into_iter()
            .map(|((first, second), v)| (format!("{}-{}", first, second), v))
            .collect();

        Ok(ScrapedExactaOdds {
            date,
            stadium_code,
            race_no,
            scraped_at: Utc::now().to_rfc3339(),
            exacta,
        })
    }

    /// Scrape trifecta odds for a single race
    pub async fn scrape_trifecta(
        &self,
        date: u32,
        stadium_code: u8,
        race_no: u8,
    ) -> Result<ScrapedTrifectaOdds, ScraperError> {
        let url = self.build_url(date, stadium_code, race_no, "trifecta");
        tracing::info!("Scraping trifecta: {}", url);

        let html = self.fetch_page(&url).await?;
        let odds = parse_trifecta_odds(&html)?;

        if odds.is_empty() {
            return Err(ScraperError::NoOddsFound);
        }

        // Expect 120 combinations (6 * 5 * 4)
        if odds.len() != 120 {
            tracing::warn!("Expected 120 combinations, got {}", odds.len());
        }

        // Convert to string keys
        let trifecta: HashMap<String, f64> = odds
            .into_iter()
            .map(|((first, second, third), v)| (format!("{}-{}-{}", first, second, third), v))
            .collect();

        Ok(ScrapedTrifectaOdds {
            date,
            stadium_code,
            race_no,
            scraped_at: Utc::now().to_rfc3339(),
            trifecta,
        })
    }

    /// Scrape all exacta odds for a stadium
    pub async fn scrape_stadium_exacta(
        &self,
        date: u32,
        stadium_code: u8,
    ) -> Vec<Result<ScrapedExactaOdds, ScraperError>> {
        let mut results = Vec::with_capacity(12);

        for race_no in 1..=12 {
            let result = self.scrape_exacta(date, stadium_code, race_no).await;
            results.push(result);
        }

        results
    }

    /// Scrape all trifecta odds for a stadium
    pub async fn scrape_stadium_trifecta(
        &self,
        date: u32,
        stadium_code: u8,
    ) -> Vec<Result<ScrapedTrifectaOdds, ScraperError>> {
        let mut results = Vec::with_capacity(12);

        for race_no in 1..=12 {
            let result = self.scrape_trifecta(date, stadium_code, race_no).await;
            results.push(result);
        }

        results
    }

    /// Scrape today's race schedule (active stadiums)
    pub async fn scrape_schedule(&self, date: u32) -> Result<TodaySchedule, ScraperError> {
        let url = format!("{}?hd={}", BASE_URL_INDEX, date);
        tracing::info!("Scraping schedule: {}", url);

        let html = self.fetch_page(&url).await?;
        parse_schedule(&html, date)
    }

    /// Scrape race entries for a single race
    pub async fn scrape_race_entries(
        &self,
        date: u32,
        stadium_code: u8,
        race_no: u8,
    ) -> Result<ScrapedRaceInfo, ScraperError> {
        let url = format!(
            "{}?rno={}&jcd={:02}&hd={}",
            BASE_URL_RACELIST, race_no, stadium_code, date
        );
        tracing::info!("Scraping entries: {}", url);

        let html = self.fetch_page(&url).await?;
        parse_race_entries(&html, date, stadium_code, race_no)
    }

    /// Scrape all race entries for a stadium
    pub async fn scrape_stadium_entries(
        &self,
        date: u32,
        stadium_code: u8,
    ) -> Vec<Result<ScrapedRaceInfo, ScraperError>> {
        let mut results = Vec::with_capacity(12);

        for race_no in 1..=12 {
            let result = self.scrape_race_entries(date, stadium_code, race_no).await;
            results.push(result);
        }

        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = ScraperConfig::default();
        assert_eq!(config.delay_ms, 2000);
        assert_eq!(config.timeout_secs, 30);
        assert_eq!(config.max_retries, 3);
    }

    #[test]
    fn test_build_url_exacta() {
        let scraper = OddsScraper::new(ScraperConfig::default());
        let url = scraper.build_url(20241230, 23, 1, "exacta");
        assert_eq!(
            url,
            "https://www.boatrace.jp/owpc/pc/race/odds2tf?rno=1&jcd=23&hd=20241230"
        );
    }

    #[test]
    fn test_build_url_trifecta() {
        let scraper = OddsScraper::new(ScraperConfig::default());
        let url = scraper.build_url(20241230, 5, 12, "trifecta");
        assert_eq!(
            url,
            "https://www.boatrace.jp/owpc/pc/race/odds3t?rno=12&jcd=05&hd=20241230"
        );
    }

    #[test]
    fn test_build_url_stadium_padding() {
        let scraper = OddsScraper::new(ScraperConfig::default());
        // Stadium code 1 should be zero-padded to 01
        let url = scraper.build_url(20241230, 1, 1, "exacta");
        assert!(url.contains("jcd=01"));
    }
}
