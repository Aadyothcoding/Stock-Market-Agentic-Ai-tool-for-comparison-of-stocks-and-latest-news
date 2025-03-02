from phi.agent import Agent
from phi.model.groq import Groq
from dotenv import load_dotenv
from phi.tools.yfinance import YFinanceTools
import feedparser

load_dotenv()

# Define the model globally to ensure no fallback to OpenAI
llama_model = Groq(id="llama-3.3-70b-versatile")

# Function to get latest stock news from Google News RSS
def get_stock_news(ticker):
    url = f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)
    
    news_list = []
    for entry in feed.entries[:5]:  # Get the top 5 news articles
        news_list.append({
            "title": entry.title,
            "link": entry.link,
            "published": entry.published
        })
    
    return news_list

# Finance Agent using YFinanceTools for stock data
finance_agent = Agent(
    model=llama_model,
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True)],
    show_tool_calls=True,
    markdown=True,
    instructions=["use tables to display data"],
    disable_fallback_to_openai=True
)

# Agent Team combining finance agent and news retrieval
def fetch_stock_data_and_news(tickers):
    results = {}
    for ticker in tickers:
        news = get_stock_news(ticker)
        results[ticker] = news
    
    finance_agent.print_response(f"Summarize and compare analyst recommendations and fundamentals for {', '.join(tickers)}")
    
    for ticker, news_list in results.items():
        print(f"\nLatest News for {ticker}:")
        for news in news_list:
            print(f"- {news['title']} ({news['published']})\n  Link: {news['link']}")

# Example usage
fetch_stock_data_and_news(["AAPL", "MSFT"])
