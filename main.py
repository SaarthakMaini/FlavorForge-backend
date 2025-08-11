from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import json
from typing import Optional, List, Dict, Any
import math
import random
from datetime import datetime, timedelta
import os

app = FastAPI(title="FlavorForge API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ProductCreate(BaseModel):
    name: str
    category: str
    ingredients: List[str]
    targetDemographic: str
    region: str
    description: Optional[str] = ""

class ProductConcept(BaseModel):
    name: str
    category: str
    ingredients: List[str]
    region: str

products_df = None
trends_df = None
metrics_df = None
competitors_df = None
analysis_df = None

def load_csv_data():
    """Load all CSV data files"""
    global products_df, trends_df, metrics_df, competitors_df, analysis_df
    
    try:
        products_df = pd.read_csv('data/products.csv')
        trends_df = pd.read_csv('data/market_trends.csv')
        metrics_df = pd.read_csv('data/dashboard_metrics.csv')
        competitors_df = pd.read_csv('data/competitors.csv')
        analysis_df = pd.read_csv('data/analysis_results.csv')
        
        print("All CSV files loaded successfully")
        print(f"Products: {len(products_df)} records")
        print(f"Market Trends: {len(trends_df)} records")
        print(f"Dashboard Metrics: {len(metrics_df)} records")
        print(f"Competitors: {len(competitors_df)} records")
        print(f"Analysis Results: {len(analysis_df)} records")
        
    except FileNotFoundError as e:
        print(f"Error loading CSV files: {e}")
        raise HTTPException(status_code=500, detail=f"CSV file not found: {e}")
    except Exception as e:
        print(f"Error loading data: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading data: {e}")

@app.on_event("startup")
async def startup_event():
    load_csv_data()

@app.get("/")
async def root():
    return {
        "message": "FlavorForge API is running",
        "version": "1.0.0",
        "endpoints": [
            "/api/dashboard/metrics",
            "/api/products",
            "/api/market-trends",
            "/api/analyze-product",
            "/api/competitors"
        ]
    }

@app.get("/api/dashboard/metrics")
async def get_dashboard_metrics(
    timeframe: str = Query("30d", description="Time period: 7d, 30d, 90d"),
    region: Optional[str] = Query(None, description="Filter by region")
):
    """Get dashboard metrics from CSV data"""
    try:
        filtered_metrics = metrics_df[metrics_df['timeframe'] == timeframe].copy()
        
        if region and region != "Global":
            filtered_metrics = filtered_metrics[
                (filtered_metrics['region'] == region) | 
                (filtered_metrics['region'] == 'Global')
            ]
        else:
            filtered_metrics = filtered_metrics[filtered_metrics['region'] == 'Global']
        
        total_products = filtered_metrics[filtered_metrics['metric_name'] == 'total_products']
        success_rate = filtered_metrics[filtered_metrics['metric_name'] == 'success_rate']
        active_users = filtered_metrics[filtered_metrics['metric_name'] == 'active_users']
        trending_categories = filtered_metrics[filtered_metrics['metric_name'] == 'trending_categories']
        
        products_growth = total_products['growth_percentage'].iloc[0] if not total_products.empty else 8.5
        success_growth = success_rate['growth_percentage'].iloc[0] if not success_rate.empty else 3.2
        users_growth = active_users['growth_percentage'].iloc[0] if not active_users.empty else 8.1
        
        metrics = {
            "totalProducts": int(total_products['metric_value'].iloc[0]) if not total_products.empty else 250,
            "successRate": float(success_rate['metric_value'].iloc[0]) if not success_rate.empty else 87.5,
            "activeUsers": int(active_users['metric_value'].iloc[0]) if not active_users.empty else 1450,
            "trendingCategories": int(trending_categories['metric_value'].iloc[0]) if not trending_categories.empty else 6,
            "growthMetrics": {
                "productsGrowth": float(products_growth),
                "successRateGrowth": float(success_growth),
                "usersGrowth": float(users_growth)
            }
        }
        
        return {"success": True, "data": metrics}
        
    except Exception as e:
        print(f"Error in get_dashboard_metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/products")
async def get_products(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(10, ge=1, le=100, description="Items per page"),
    category: Optional[str] = Query(None, description="Filter by category"),
    search: Optional[str] = Query(None, description="Search in product names"),
    sort_by: str = Query("name", description="Sort by field")
):
    """Get products with filtering, search, and pagination"""
    try:
        df = products_df.copy()
        
        if search:
            search_mask = (
                df['name'].str.contains(search, case=False, na=False) |
                df['category'].str.contains(search, case=False, na=False) |
                df['ingredients'].str.contains(search, case=False, na=False)
            )
            df = df[search_mask]

        if category and category != "":
            df = df[df['category'].str.contains(category, case=False, na=False)]

        if sort_by in df.columns:
            df = df.sort_values(by=sort_by)

        total = len(df)
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        paginated_df = df.iloc[start_idx:end_idx]

        products = []
        for _, row in paginated_df.iterrows():
            ingredients = row.get('ingredients', '')
            if isinstance(ingredients, str):
                ingredients = [ing.strip() for ing in ingredients.split(',') if ing.strip()]
            
            products.append({
                "id": str(row['id']),
                "name": row['name'],
                "category": row['category'],
                "ingredients": ingredients,
                "marketScore": int(row.get('market_score', 85)),
                "region": row.get('region', 'Global'),
                "targetDemographic": row.get('target_demographics', 'General'),
                "createdDate": row.get('created_date', '2024-01-15'),
                "status": row.get('status', 'active').lower()
            })
        
        return {
            "success": True,
            "data": {
                "products": products,
                "total": total,
                "page": page,
                "totalPages": math.ceil(total / limit) if total > 0 else 1
            }
        }
        
    except Exception as e:
        print(f"Error in get_products: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/products")
async def create_product(product_data: ProductCreate):
    """Create a new product with AI-calculated market score and save to CSV"""
    try:
        global products_df

        score = calculate_market_score(product_data.ingredients, product_data.region)

        max_id = products_df['id'].max() if not products_df.empty else 0
        new_id = max_id + 1

        new_product = {
            "id": int(new_id),
            "name": product_data.name,
            "category": product_data.category,
            "ingredients": ", ".join(product_data.ingredients),
            "market_score": score,
            "region": product_data.region,
            "target_demographics": product_data.targetDemographic,
            "created_date": datetime.now().strftime("%Y-%m-%d"),
            "status": "testing"
        }

        products_df = pd.concat([products_df, pd.DataFrame([new_product])], ignore_index=True)
        products_df.to_csv("data/products.csv", index=False)
        load_csv_data()

        return {"success": True, "data": new_product}

    except Exception as e:
        print(f"Error in create_product: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/market-trends")
async def get_market_trends(
    region: Optional[str] = Query(None, description="Filter by region"),
    category: Optional[str] = Query(None, description="Filter by category")
):
    """Get market trends with filtering"""
    try:
        df = trends_df.copy()
        
        if region and region != "":
            region_mask = (
                df['region'].str.contains(region, case=False, na=False) |
                df['region'].str.contains('Global', case=False, na=False)
            )
            df = df[region_mask]
        
        if category and category != "":
            df = df[df['category'].str.contains(category, case=False, na=False)]

        trends = []
        for _, row in df.iterrows():
            trends.append({
                "id": str(row['id']),
                "ingredient": row.get('ingredient_name', row.get('ingredient', 'Unknown')),
                "popularity": int(row.get('popularity_score', row.get('popularity', 80))),
                "growthRate": float(row.get('growth_rate', row.get('growthRate', 10.0))),
                "region": row.get('region', 'Global'),
                "category": row.get('category', 'General')
            })
        
        return {"success": True, "data": trends}
        
    except Exception as e:
        print(f"Error in get_market_trends: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze-product")
async def analyze_product(product_concept: ProductConcept):
    """Analyze product concept using AI and market data"""
    try:
        overall_score = calculate_market_score(product_concept.ingredients, product_concept.region)
        market_potential = min(100, overall_score + random.randint(-10, 15))
        trend_alignment = calculate_trend_alignment(product_concept.ingredients)
        
        category_competitors = competitors_df[
            competitors_df['primary_category'].str.contains(product_concept.category, case=False, na=False)
        ]
        
        competitor_count = len(category_competitors)
        avg_market_share = category_competitors['market_share'].mean() if not category_competitors.empty else 10.0

        recommendations = generate_recommendations(
            product_concept.ingredients, 
            product_concept.category,
            overall_score
        )
        
        analysis = {
            "overallScore": overall_score,
            "marketPotential": market_potential,
            "competitiveAnalysis": {
                "competitors": competitor_count,
                "marketShare": round(avg_market_share / competitor_count if competitor_count > 0 else 5.0, 1)
            },
            "recommendations": recommendations,
            "trendAlignment": trend_alignment
        }
        
        return {"success": True, "data": analysis}
        
    except Exception as e:
        print(f"Error in analyze_product: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/competitors")
async def get_competitors(category: Optional[str] = Query(None, description="Filter by category")):
    """Get competitor analysis"""
    try:
        df = competitors_df.copy()

        if category and category != "":
            df = df[df['primary_category'].str.contains(category, case=False, na=False)]

        df = df.sort_values('market_share', ascending=False)

        competitors = []
        for _, row in df.iterrows():
            competitors.append({
                "id": str(row['id']),
                "name": row['company_name'],
                "marketShare": float(row['market_share']),
                "products": int(row['total_products']),
                "category": row['primary_category']
            })
        
        return {"success": True, "data": competitors}
        
    except Exception as e:
        print(f"Error in get_competitors: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def calculate_market_score(ingredients: List[str], region: str) -> int:
    """Calculate market score based on ingredient trends and regional preferences"""
    try:
        base_score = 75
        trend_bonus = 0
        
        for ingredient in ingredients:
            ingredient_trends = trends_df[
                trends_df['ingredient_name'].str.contains(ingredient, case=False, na=False) |
                trends_df.get('ingredient', pd.Series()).str.contains(ingredient, case=False, na=False)
            ]
            
            if not ingredient_trends.empty:
                avg_popularity = ingredient_trends['popularity_score'].mean() if 'popularity_score' in ingredient_trends.columns else ingredient_trends.get('popularity', pd.Series([80])).mean()
                avg_growth = ingredient_trends['growth_rate'].mean() if 'growth_rate' in ingredient_trends.columns else ingredient_trends.get('growthRate', pd.Series([10])).mean()
                
                popularity_bonus = (avg_popularity - 70) * 0.2
                growth_bonus = avg_growth * 0.1
                trend_bonus += popularity_bonus + growth_bonus
        
        regional_bonus = 0
        if region in ['Asia Pacific', 'Asia-Pacific']:
            regional_bonus = 5
        elif region == 'North America':
            regional_bonus = 3
        
        final_score = min(100, max(60, base_score + trend_bonus + regional_bonus + random.randint(-5, 10)))
        return int(final_score)
        
    except Exception as e:
        print(f"Error calculating market score: {e}")
        return 80

def calculate_trend_alignment(ingredients: List[str]) -> int:
    """Calculate how well ingredients align with current trends"""
    try:
        alignment_score = 80
        
        for ingredient in ingredients:
            ingredient_trends = trends_df[
                trends_df['ingredient_name'].str.contains(ingredient, case=False, na=False) |
                trends_df.get('ingredient', pd.Series()).str.contains(ingredient, case=False, na=False)
            ]
            
            if not ingredient_trends.empty:
                avg_growth = ingredient_trends['growth_rate'].mean() if 'growth_rate' in ingredient_trends.columns else ingredient_trends.get('growthRate', pd.Series([10])).mean()
                
                if avg_growth > 20:
                    alignment_score += 8
                elif avg_growth > 15:
                    alignment_score += 5
                elif avg_growth > 10:
                    alignment_score += 2
        
        return min(100, max(60, alignment_score + random.randint(-5, 5)))
        
    except Exception as e:
        print(f"Error calculating trend alignment: {e}")
        return 85

def generate_recommendations(ingredients: List[str], category: str, score: int) -> List[str]:
    """Generate AI recommendations based on product analysis"""
    recommendations = []

    if score >= 90:
        recommendations.append("Excellent market potential - proceed with full development")
        recommendations.append("Consider premium positioning strategy")
    elif score >= 80:
        recommendations.append("Strong market opportunity with careful positioning")
        recommendations.append("Focus on ingredient benefits messaging")
    elif score >= 70:
        recommendations.append("Moderate potential - consider reformulation with trending ingredients")
        recommendations.append("Target specific consumer segments")
    else:
        recommendations.append("Requires significant market education")
        recommendations.append("Consider cost-effective production methods")
    
    if category.lower() == 'beverages':
        recommendations.append("Leverage social media marketing for beverage launches")
        recommendations.append("Consider seasonal marketing campaigns")
    elif category.lower() == 'snacks':
        recommendations.append("Focus on convenience and portability messaging")
        recommendations.append("Target health-conscious millennials")
    elif category.lower() == 'dairy':
        recommendations.append("Emphasize natural and organic positioning")
        recommendations.append("Consider plant-based alternatives")
    
    trending_ingredients = ['matcha', 'turmeric', 'acai', 'spirulina', 'quinoa']
    has_trending = any(ing.lower() in ' '.join(ingredients).lower() for ing in trending_ingredients)
    
    if has_trending:
        recommendations.append("Capitalize on superfood trend positioning")
    else:
        recommendations.append("Consider adding trending superfood ingredients")
    
    return recommendations[:3]

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "data_loaded": all([
            products_df is not None,
            trends_df is not None,
            metrics_df is not None,
            competitors_df is not None,
            analysis_df is not None
        ])
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)