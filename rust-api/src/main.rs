use actix_web::{web, App, HttpServer, middleware};
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

mod handlers;
mod models;
mod predictor;

use handlers::{health, predict};

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // Initialize logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)
        .expect("Failed to set subscriber");

    let host = std::env::var("HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
    let port = std::env::var("PORT").unwrap_or_else(|_| "8080".to_string());
    let addr = format!("{}:{}", host, port);

    info!("Starting Boatrace API server at http://{}", addr);

    HttpServer::new(|| {
        App::new()
            .wrap(middleware::Logger::default())
            .route("/health", web::get().to(health::health_check))
            .route("/predict", web::post().to(predict::predict_race))
            .route("/predict/exacta", web::post().to(predict::predict_exacta))
    })
    .bind(&addr)?
    .run()
    .await
}
