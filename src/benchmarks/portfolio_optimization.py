"""
Utility-Based Portfolio Optimization Benchmark for AgenticEvals.

This benchmark tests a model's ability to act as a utility-based agent
for a financial task. The agent must allocate a starting capital on an asset from
a set of assets to maximize expected profit for a single upcoming time period,
based on a "market forecast" presented as a news article.

The model must:
1.  Parse a news article containing a mix of relevant financial predictions
    and irrelevant distractor information.
2.  Identify the true expected price changes for each asset.
3.  Calculate the optimal allocation of capital to maximize expected utility (profit),
    considering transaction costs.
4.  Generate a list of trades in the specified format.
"""

import time
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict

from ..benchmark.base import BaseBenchmark, Task, TaskResult, BenchmarkConfig, AgentType
from ..benchmark.registry import benchmark
from ..models.base import BaseModel, ModelResponse
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Asset:
    """Represents a financial asset available for trading."""
    id: str
    name: str
    current_price: float
    # The true, hidden forecast for the solver to use.
    expected_price: float

@dataclass
class Trade:
    """Represents a single trade action."""
    asset_id: str
    shares: int # Positive for buy, could be negative for sell if allowed

@dataclass
class Portfolio:
    """Represents the agent's proposed portfolio of trades."""
    trades: List[Trade]

    def __str__(self):
        return ", ".join([f"BUY {t.shares} {t.asset_id}" for t in self.trades])


@benchmark(
    name="portfolio_optimization",
    agent_type=AgentType.UTILITY_BASED,
    description="Allocates capital across assets to maximize expected utility based on a market forecast."
)
class PortfolioOptimizationBenchmark(BaseBenchmark):
    """
    Benchmark for utility-based portfolio optimization. The agent must parse
    a news report, extract actionable intelligence, and allocate capital to
    maximize expected returns.
    """

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)

    def get_tasks(self) -> List[Task]:
        """Get all tasks for the portfolio optimization benchmark."""
        tasks = []
        scenarios = self._get_scenarios()
        for i, scenario in enumerate(scenarios):
            optimal_portfolio, optimal_utility = self._solve_portfolio(
                scenario['assets'],
                scenario['capital'],
                scenario['transaction_cost_per_trade']
            )
            task = Task(
                task_id=f"portfolio_optimization_{i+1}",
                name=f"Portfolio Optimization: {scenario['name']}",
                description=scenario['description'],
                prompt=self._create_prompt(scenario),
                expected_output=str(optimal_portfolio),
                metadata={
                    "scenario": scenario,
                    "optimal_utility": optimal_utility,
                    "optimal_portfolio": str(optimal_portfolio),
                    "difficulty": scenario['difficulty'],
                }
            )
            tasks.append(task)
        return tasks

    def _get_scenarios(self) -> List[Dict[str, Any]]:
        """Defines the scenarios for the benchmark, complete with distractor news."""
        return [
            # Scenario 1: Simple, clear winner
            {
                "name": "Clear Winner",
                "description": "A straightforward scenario with one obviously profitable asset.",
                "difficulty": "simple",
                "capital": 10000.0,
                "transaction_cost_per_trade": 10.0,
                "assets": [
                    Asset(id="STKA", name="Stock A", current_price=100.0, expected_price=115.0),
                    Asset(id="STKB", name="Stock B", current_price=50.0, expected_price=50.0),
                    Asset(id="STKC", name="Stock C", current_price=200.0, expected_price=200.0),
                    Asset(id="STKD", name="Stock D", current_price=30.0, expected_price=35.0),
                    Asset(id="STKE", name="Stock E", current_price=120.0, expected_price=120.0),
                    Asset(id="STKF", name="Stock F", current_price=90.0, expected_price=90.0),
                ],
                "news": (
                    "**SPORTS DIGEST:**\n"
                    "The City A Eagles secured a surprising 28-24 victory over the City B Bears last night, a result that shocked fans and analysts alike. "
                    "Eagles quarterback, Mike Swanson, threw for three touchdowns. The local university's rowing team also won their regional championship. "
                    "The annual City A marathon has been postponed due to a heatwave advisory, and a local artist just unveiled a new sculpture in the city park.\n\n"
                    "**CELEBRITY GOSSIP:**\n"
                    "Pop star 'Glitter' announced that her upcoming album, 'Electric Dreams', will be delayed by three months due to production issues. Her last album sold over 10 million copies worldwide. "
                    "A new fashion trend, 'utility vests', is sweeping the nation.\n\n"
                    "**FINANCIAL TIMES:**\n"
                    "In business news, a market analyst from Bank A suggested that Stock A (STKA) is poised for a significant rally. The analyst cited strong Q2 earnings and a new product launch, "
                    "predicting a target price of $115.00 in the near term. This represents a significant upside from its current valuation. "
                    "Meanwhile, the government announced that 10-year bonds are currently yielding 2 percent. "
                    "Stock B (STKB), a major player in the agricultural sector, is expected to remain stable, with analysts seeing neither strong headwinds nor tailwinds following a government report on crop yields. "
                    "The CEO of Stock C (STKC), a large retail chain, announced they are maintaining their current dividend payout and have no major announcements planned. "
                    "A tech startup, Stock D (STKD), has just received a major new patent and analysts predict its price will jump to $35. "
                    "Stock E (STKE) and Stock F (STKF) are expected to remain flat."
                ),
            },
            # Scenario 2: More distractors
            {
                "name": "Noisy News",
                "description": "One profitable asset hidden within irrelevant news.",
                "difficulty": "medium",
                "capital": 20000.0,
                "transaction_cost_per_trade": 20.0,
                "assets": [
                    Asset(id="STKA", name="Stock A", current_price=250.0, expected_price=245.0),
                    Asset(id="STKB", name="Stock B", current_price=80.0, expected_price=95.0),
                    Asset(id="STKC", name="Stock C", current_price=120.0, expected_price=120.0),
                    Asset(id="STKD", name="Stock D", current_price=300.0, expected_price=300.0),
                ],
                "news": (
                    "**NATIONAL GEOGRAPHIC:**\n"
                    "A new documentary on the migration patterns of the arctic tern is set to be released next month. The film crew spent over a year in remote locations capturing stunning footage. "
                    "The city's public library has announced a new 'read-a-thon' event for charity.\n\n"
                    "**TRAFFIC & TRANSPORTATION:**\n"
                    "Main Street will be closed for a parade this Saturday between 10 AM and 2 PM. Motorists are advised to seek alternate routes. "
                    "A new bike lane was approved by the city council.\n\n"
                    "**GLOBAL ECONOMIC OUTLOOK:**\n"
                    "The government of Country B announced new tariffs on imported steel, a move that has drawn criticism from its trading partners. "
                    "The annual city marathon is scheduled for next month, which is expected to bring a surge in tourism and revenue for local businesses. "
                    "The national inflation rate was reported at 3 percent this morning, a slight increase from the previous quarter. A report from a leading agricultural consultancy "
                    "highlighted strong global demand for Stock B's (STKB) products, projecting a price of $95.00. Stock A (STKA) "
                    "is facing headwinds from the new steel tariffs, with some seeing a slight dip to $245.00. Stock C's (STKC) dividend yield is currently 4 percent, and the weather forecast predicts a sunny weekend for the marathon. "
                    "Stock D (STKD), a leader in the software industry, is not expected to see significant movement as it prepares for a minor version update to its flagship product."
                ),
            },
            # Scenario 3: The Transaction Cost Trap
            {
                "name": "Transaction Cost Trap",
                "description": "An asset appears profitable, but the gain is less than the transaction cost.",
                "difficulty": "medium",
                "capital": 5000.0,
                "transaction_cost_per_trade": 25.0,
                "assets": [
                    Asset(id="STKA", name="Stock A", current_price=50.0, expected_price=50.20), # 20 cent gain per share
                    Asset(id="STKB", name="Stock B", current_price=180.0, expected_price=175.0),
                    Asset(id="STKC", name="Stock C", current_price=75.0, expected_price=75.0),
                    Asset(id="STKD", name="Stock D", current_price=20.0, expected_price=25.0),
                    Asset(id="STKE", name="Stock E", current_price=10.0, expected_price=10.0),
                    Asset(id="STKF", name="Stock F", current_price=5.0, expected_price=5.0),
                ],
                "news": (
                    "**CITY COUNCIL MINUTES:**\n"
                    "The City C council approved a new zoning law for residential areas after a lengthy three-hour debate. The vote was close, passing 5-4. "
                    "Separately, the parks department announced the planting of 50 new oak trees. The city's annual chili cook-off has been scheduled for August 5th.\n\n"
                    "**MUSIC CHARTS:**\n"
                    "The indie band 'The Ferrets' has reached number one on the streaming charts with their new single 'Sunset Drive'. The band announced a new tour to celebrate. "
                    "Vinyl record sales have seen a 15% increase this year.\n\n"
                    "**ENERGY SECTOR REPORT:**\n"
                    "Stock B (STKB), an oil exploration company, announced lower-than-expected drilling results from their flagship site, with analysts at 'Petro Insights' revising their price target down to $175. "
                    "The company also announced a stock split for an unrelated subsidiary focused on renewable energy. "
                    "A separate report mentioned that Stock A (STKA), a microchip manufacturer, is expected to see a marginal price increase to $50.20 due to a minor patent approval for a new capacitor design. The local football team, the 'City C Cyclones', is on a surprising winning streak. "
                    "Stock C (STKC) is a utility company that recently completed a planned infrastructure upgrade, and its price is expected to remain stable. "
                    "A new solar company, Stock D (STKD), just announced a major government contract and is expected to rise to $25. "
                    "Stock E (STKE) and Stock F (STKF) are not expected to see any price changes."
                ),
            },
            # Scenario 4: All Unprofitable
            {
                "name": "All Unprofitable",
                "description": "No assets are expected to be profitable; the correct action is to hold.",
                "difficulty": "hard",
                "capital": 100000.0,
                "transaction_cost_per_trade": 50.0,
                "assets": [
                    Asset(id="STKA", name="Stock A", current_price=500.0, expected_price=490.0),
                    Asset(id="STKB", name="Stock B", current_price=30.0, expected_price=30.0),
                    Asset(id="STKC", name="Stock C", current_price=88.0, expected_price=88.0),
                ],
                "news": (
                    "**CULTURAL REVIEW:**\n"
                    "The national museum in Country D announced a new exhibit on ancient history, which will feature artifacts from a recent archaeological dig. The exhibit is expected to run for six months. "
                    "A new biography about a famous historical figure has just been released and is receiving critical acclaim.\n\n"
                    "**WEATHER REPORT:**\n"
                    "A mild winter is being predicted for the region, which could impact sales of winter clothing. "
                    "Next week's forecast is mostly sunny with a small chance of rain on Friday.\n\n"
                    "**MARKET PULSE:**\n"
                    "A global economic slowdown is impacting industrial sectors. Stock A (STKA) is expected to dip to $490 amidst the downturn in manufacturing. "
                    "Their main competitor, 'Massive Corp', is also expected to see a 5 percent drop in share price. "
                    "Stock B (STKB), a consumer staples company, is predicted to show no change in its valuation as demand for its products remains inelastic. "
                    "Telecom company Stock C (STKC) has announced it will be maintaining its current network infrastructure with no new capital expenditures planned for the next quarter."
                ),
            },
            # Scenario 5: Multiple profitable assets, find the best one
            {
                "name": "Finding the Best Opportunity",
                "description": "Multiple assets are profitable, but one offers a clearly superior return on investment.",
                "difficulty": "hard",
                "capital": 50000.0,
                "transaction_cost_per_trade": 15.0,
                "assets": [
                    Asset(id="STKA", name="Stock A", current_price=100.0, expected_price=105.0),
                    Asset(id="STKB", name="Stock B", current_price=20.0, expected_price=24.0),
                    Asset(id="STKC", name="Stock C", current_price=400.0, expected_price=410.0),
                    Asset(id="STKD", name="Stock D", current_price=150.0, expected_price=150.0),
                    Asset(id="STKE", name="Stock E", current_price=15.0, expected_price=20.0),
                    Asset(id="STKF", name="Stock F", current_price=80.0, expected_price=80.0),
                    Asset(id="STKG", name="Stock G", current_price=50.0, expected_price=50.0),
                ],
                "news": (
                    "**LOCAL ENTERTAINMENT:**\n"
                    "A local film festival in City E is drawing large crowds, with several independent films receiving rave reviews. The event has been a boon for local restaurants. "
                    "A new play is opening at the downtown theater next month.\n\n"
                    "**FASHION WEEK DAILY:**\n"
                    "This season's must-have color is 'burnt orange', according to top designers. Sales of vintage clothing are also reportedly on the rise. "
                    "A famous model was seen wearing a new brand of sneakers.\n\n"
                    "**INVESTOR'S CHRONICLE:**\n"
                    "Market analysis is varied this week. A report from InvestRight suggests Stock C (STKC), a major utility company, will climb to $410. "
                    "Their debt was recently rated as investment grade. A separate brief from Capital Gains Weekly projects that Stock B (STKB), a biotech firm, will jump to $24 based on a successful new product launch. "
                    "Stock A (STKA), a blue-chip industrial, is also seen as a safe bet, with predictions of a modest rise to $105. "
                    "Stock D (STKD), a major shipping company, reported earnings in line with expectations and its stock is not expected to move. "
                    "A new gaming company, Stock E (STKE), is expected to see a significant jump to $20 after a hit new release. "
                    "Stock F (STKF) and Stock G (STKG) are not expected to see any price changes."
                ),
            },
            # Scenario 6: Conflicting Reports
            {
                "name": "Conflicting Reports",
                "description": "Two analysts offer conflicting advice on the same stock, requiring the agent to pick the more credible source.",
                "difficulty": "very_hard",
                "capital": 25000.0,
                "transaction_cost_per_trade": 20.0,
                "assets": [
                    Asset(id="STKA", name="Stock A", current_price=300.0, expected_price=330.0),
                    Asset(id="STKB", name="Stock B", current_price=150.0, expected_price=150.0),
                    Asset(id="STKC", name="Stock C", current_price=500.0, expected_price=500.0),
                ],
                "news": (
                    "**SOCIETY PAGES:**\n"
                    "City F is preparing for its annual Founder's Day parade, a tradition that dates back over a century. A new float has been designed by a local artist. "
                    "The historical society is seeking volunteers for a new archiving project.\n\n"
                    "**TECHNOLOGY TODAY:**\n"
                    "A new social media app, 'Chirp', is gaining popularity among teenagers. It features short video clips and has over 5 million downloads already. "
                    "Sales of virtual reality headsets are up 20% this quarter.\n\n"
                    "**WALL STREET WHISPERS:**\n"
                    "Financial news is abuzz with conflicting reports on Stock A (STKA). Analyst A from 'Future Finance', a highly-rated firm with a 95 percent accuracy rating, projects a strong buy with a price target of $330, citing a new technology patent. However, Analyst B from 'QuickBucks Analysis', a firm known for sensationalist claims and a 30 percent accuracy rating, downgraded the stock to 'hold', mentioning potential regulatory hurdles. "
                    "In unrelated news, a different company, 'Data Corp', just reported a 15 percent increase in quarterly earnings. Stock B (STKB) is expected to remain flat. "
                    "Luxury car maker Stock C (STKC) just announced that their sales for the last quarter were exactly as predicted."
                )
            },
            # Scenario 7: High Volume, Low Margin
            {
                "name": "High Volume, Low Margin",
                "description": "An asset has a very small profit margin, requiring a large investment to be worthwhile.",
                "difficulty": "hard",
                "capital": 1000000.0,
                "transaction_cost_per_trade": 5.0,
                "assets": [
                    Asset(id="STKA", name="Stock A", current_price=10.0, expected_price=10.01),
                    Asset(id="STKB", name="Stock B", current_price=1000.0, expected_price=950.0),
                    Asset(id="STKC", name="Stock C", current_price=20.0, expected_price=20.0),
                    Asset(id="STKD", name="Stock D", current_price=5.0, expected_price=5.01),
                    Asset(id="STKE", name="Stock E", current_price=2.0, expected_price=2.0),
                    Asset(id="STKF", name="Stock F", current_price=1.0, expected_price=1.0),
                ],
                "news": (
                    "**LITERARY REVIEW:**\n"
                    "A famous author announced a book signing tour in several cities across Country G, following the release of their new best-selling novel. The tour is expected to be a major cultural event. "
                    "Sales of e-books have grown by 8% this year, while physical book sales have remained flat.\n\n"
                    "**HEALTH & LIFESTYLE:**\n"
                    "A new fitness trend involving weighted hula hoops is becoming popular in major cities. "
                    "A recent study has shown that drinking green tea may have moderate health benefits.\n\n"
                    "**COMMODITIES & TRADE:**\n"
                    "A new trade agreement is expected to provide a minor but steady boost to Stock A (STKA), with projections showing a slight increase to $10.01. "
                    "The consumer price index rose by 0.2 percent last month, slightly below expectations. "
                    "Stock B (STKB), a luxury goods retailer, lost a major contract, and its stock is expected to fall to $950. "
                    "The beverage company Stock C (STKC) is not expected to see any price change this quarter. "
                    "A new penny stock, Stock D (STKD), is expected to see a tiny increase to $5.01. "
                    "Stock E (STKE) and Stock F (STKF) are stable."
                )
            },
            # Scenario 8: The Decoy
            {
                "name": "The Decoy",
                "description": "A seemingly great opportunity is mentioned, but for a stock that is not on the available list.",
                "difficulty": "hard",
                "capital": 15000.0,
                "transaction_cost_per_trade": 10.0,
                "assets": [
                    Asset(id="STKA", name="Stock A", current_price=75.0, expected_price=78.0),
                    Asset(id="STKB", name="Stock B", current_price=90.0, expected_price=90.0),
                    Asset(id="STKC", name="Stock C", current_price=110.0, expected_price=110.0),
                ],
                "news": (
                    "**PUBLIC BROADCASTING ANNOUNCEMENT:**\n"
                    "The local library in City H has extended its hours to 9 PM on weekdays, a move celebrated by community leaders. The change was made possible by a recent philanthropic donation. "
                    "The annual bake sale for the local school district raised a record amount of funds this year.\n\n"
                    "**AUTOMOTIVE NEWS:**\n"
                    "A new electric pickup truck was just announced by a major car manufacturer, with impressive specs and a competitive price. "
                    "The price of gasoline has risen by 5% in the last month.\n\n"
                    "**CONSTRUCTION TODAY:**\n"
                    "The government announced a massive infrastructure spending bill. This has sent shares of a construction giant, 'BuildItAll' (Ticker: BLD), soaring, with analysts predicting a 25 percent jump. "
                    "Meanwhile, Stock A (STKA), a related but smaller company that supplies materials to BLD, is expected to see a modest gain to $78. "
                    "The CEO of Stock B (STKB), a software company, is retiring after a successful 20-year tenure. "
                    "Stock C (STKC), a food processing company, has reported stable earnings for the third consecutive quarter."
                )
            },
            # Scenario 9: Qualitative to Quantitative
            {
                "name": "Qualitative to Quantitative",
                "description": "The forecast is given in qualitative terms, requiring the agent to infer a reasonable quantitative value.",
                "difficulty": "very_hard",
                "capital": 30000.0,
                "transaction_cost_per_trade": 30.0,
                "assets": [
                    Asset(id="STKA", name="Stock A", current_price=200.0, expected_price=220.0),
                    Asset(id="STKB", name="Stock B", current_price=400.0, expected_price=400.0),
                    Asset(id="STKC", name="Stock C", current_price=50.0, expected_price=50.0),
                    Asset(id="STKD", name="Stock D", current_price=25.0, expected_price=30.0),
                    Asset(id="STKE", name="Stock E", current_price=60.0, expected_price=60.0),
                    Asset(id="STKF", name="Stock F", current_price=70.0, expected_price=70.0),
                ],
                "news": (
                    "**HEALTH & WELLNESS JOURNAL:**\n"
                    "A new study published in the National Health Journal on the benefits of drinking at least eight glasses of water per day has gained widespread media attention. "
                    "The city just opened a new public swimming pool.\n\n"
                    "**ENTERTAINMENT WEEKLY:**\n"
                    "A historical drama film is leading the box office for the third week in a row. "
                    "A popular TV series was just renewed for a fourth season.\n\n"
                    "**PHARMACEUTICAL NEWS:**\n"
                    "A major breakthrough in medical research was announced by Stock A (STKA). An industry expert stated, 'This is a game-changer for their entire product pipeline; I expect to see at least a 10 percent uplift in their stock price very soon.' "
                    "Stock B (STKB), a competitor, continues its stable performance. Their new drug trial is proceeding to phase 2, with results expected next year. "
                    "Stock C (STKC), a company that manufactures generic drugs, is not expected to see any major price changes. "
                    "A new fitness company, Stock D (STKD), is projected to rise to $30 after partnering with a major celebrity. "
                    "Stock E (STKE) and Stock F (STKF) are holding steady."
                )
            },
            # Scenario 10: Multi-step Reasoning
            {
                "name": "Multi-step Reasoning",
                "description": "The agent must connect two pieces of news to understand the full impact on a stock.",
                "difficulty": "very_hard",
                "capital": 40000.0,
                "transaction_cost_per_trade": 25.0,
                "assets": [
                    Asset(id="STKA", name="Stock A", current_price=150.0, expected_price=165.0),
                    Asset(id="STKB", name="Stock B", current_price=50.0, expected_price=50.0),
                    Asset(id="STKC", name="Stock C", current_price=120.0, expected_price=120.0),
                    Asset(id="STKD", name="Stock D", current_price=40.0, expected_price=48.0),
                    Asset(id="STKE", name="Stock E", current_price=30.0, expected_price=30.0),
                    Asset(id="STKF", name="Stock F", current_price=20.0, expected_price=20.0),
                ],
                "news": (
                    "**CITY GAZETTE:**\n"
                    "The city government of City J has approved a new ride-sharing service, 'GoFast', which will launch next month. Analysts expect this to significantly increase the demand for new, economical vehicles in the metropolitan area. The weather is forecast to be mild this week. "
                    "The city also announced a plan to repave several major roads over the summer.\n\n"
                    "**FOOD & DINING:**\n"
                    "A new restaurant specializing in fusion cuisine has opened downtown and is receiving rave reviews from critics. "
                    "A recent survey shows that coffee consumption is up 5% nationally.\n\n"
                    "**AUTO INSIDER:**\n"
                    "Stock A (STKA) has just launched a new, affordable electric car model that is perfect for fleet services and high-mileage drivers. "
                    "Separately, Stock B (STKB), a company specializing in heavy-duty trucks, recently restructured its debt, a move seen as positive for its long-term stability but with no immediate market impact. "
                    "Stock C (STKC), a company that makes car batteries, is expected to maintain its current price. "
                    "A parts supplier, Stock D (STKD), is expected to see a jump to $48 due to the increased demand from GoFast. "
                    "Stock E (STKE) and Stock F (STKF) are not in the auto industry and will not be affected."
                )
            },
            # Scenario 11: The Time Horizon Trap
            {
                "name": "The Time Horizon Trap",
                "description": "A positive prediction is given, but for a long-term horizon that is irrelevant to the single-period task.",
                "difficulty": "hard",
                "capital": 60000.0,
                "transaction_cost_per_trade": 40.0,
                "assets": [
                    Asset(id="STKA", name="Stock A", current_price=500.0, expected_price=500.0),
                    Asset(id="STKB", name="Stock B", current_price=100.0, expected_price=102.0),
                    Asset(id="STKC", name="Stock C", current_price=300.0, expected_price=300.0),
                    Asset(id="STKD", name="Stock D", current_price=25.0, expected_price=25.0),
                    Asset(id="STKE", name="Stock E", current_price=55.0, expected_price=55.0),
                    Asset(id="STKF", name="Stock F", current_price=70.0, expected_price=70.0),
                ],
                "news": (
                    "**ART & CULTURE WEEKLY:**\n"
                    "The national art gallery in Country K opened a new wing dedicated to modernist sculpture. The opening was attended by several high-profile patrons. "
                    "The city's symphony orchestra announced its schedule for the upcoming season. "
                    "Sales of tickets for the annual film festival are in line with last year's figures.\n\n"
                    "**TRAVEL & LEISURE:**\n"
                    "A new report indicates a growing trend of 'staycations', where people take vacations in their home country. "
                    "A popular travel blogger just released a guide to the best beaches in the region. "
                    "Hotel chain Stock E (STKE) reported that its booking numbers are stable.\n\n"
                    "**CEO SPOTLIGHT:**\n"
                    "The visionary CEO of Stock A (STKA) laid out a 10-year strategic plan that analysts believe could double the company's value. The plan focuses on expanding into new international markets. However, in the short term, no significant price movement is expected as the plan requires substantial long-term investment. The company's 5-year bonds are currently yielding 4 percent. "
                    "In other news, Stock B (STKB) received a small government subsidy for its green manufacturing initiatives, which is likely to cause a slight bump to $102. "
                    "Stock C (STKC) has a stable outlook, with analysts not predicting any significant changes. "
                    "Stock D (STKD), a popular coffee chain, is also expected to hold its value. "
                    "Stock F (STKF), a renewable energy company, is holding a press conference next week, but analysts do not expect any market-moving announcements."
                )
            },
            # Scenario 12: High Risk, High Reward
            {
                "name": "High Risk, High Reward",
                "description": "A very volatile stock has a high potential upside but also a chance of a significant loss. The agent must weigh the expected value.",
                "difficulty": "very_hard",
                "capital": 20000.0,
                "transaction_cost_per_trade": 15.0,
                "assets": [
                    Asset(id="STKA", name="Stock A", current_price=100.0, expected_price=110.0),
                    Asset(id="STKB", name="Stock B", current_price=80.0, expected_price=81.0),
                    Asset(id="STKC", name="Stock C", current_price=95.0, expected_price=95.0),
                    Asset(id="STKD", name="Stock D", current_price=120.0, expected_price=120.0),
                    Asset(id="STKE", name="Stock E", current_price=200.0, expected_price=200.0),
                    Asset(id="STKF", name="Stock F", current_price=30.0, expected_price=30.0),
                ],
                "news": (
                    "**FOOD & DINING:**\n"
                    "A local bakery in City L won a prestigious national prize for its artisanal croissants. The bakery has seen its morning queues triple in length since the announcement. "
                    "A new study suggests that dark chocolate may have antioxidant benefits.\n\n"
                    "**SPORTS UPDATE:**\n"
                    "The local basketball team is on a 5-game winning streak. Their next game is on Tuesday. "
                    "A famous athlete just announced their retirement. Stock D (STKD) is a sports betting company whose stock is not expected to move.\n\n"
                    "**LEGAL & BUSINESS DAILY:**\n"
                    "Stock A (STKA) is awaiting a final court ruling on a major patent infringement case. Analysts from 'Risk Takers Inc.' say there is a 60 percent chance of a win, which would send the stock soaring to $150, and a 40 percent chance of a loss, which would see it plummet to $50. "
                    "Meanwhile, Stock B (STKB) is expected to inch up to $81. The CEO of Stock B just won a community service award for their volunteer work. "
                    "Stock C (STKC), a clothing retailer, is not expected to see its valuation change this quarter. "
                    "Stock E (STKE), a major airline, has a stable outlook. "
                    "Stock F (STKF), a streaming service, is facing stiff competition and is not expected to grow."
                )
            },
            # Scenario 13: The Hidden Gem
            {
                "name": "The Hidden Gem",
                "description": "A small, overlooked detail in a long report points to a profitable trade.",
                "difficulty": "hard",
                "capital": 10000.0,
                "transaction_cost_per_trade": 10.0,
                "assets": [
                    Asset(id="STKA", name="Stock A", current_price=40.0, expected_price=40.0),
                    Asset(id="STKB", name="Stock B", current_price=60.0, expected_price=65.0),
                    Asset(id="STKC", name="Stock C", current_price=130.0, expected_price=130.0),
                    Asset(id="STKD", name="Stock D", current_price=85.0, expected_price=82.0),
                    Asset(id="STKE", name="Stock E", current_price=210.0, expected_price=210.0),
                    Asset(id="STKF", name="Stock F", current_price=12.0, expected_price=12.0),
                ],
                "news": (
                    "**GOVERNMENT PRESS RELEASE:**\n"
                    "A lengthy 200-page government report on national logistics was released today in Country M. Most of the media focus has been on the shipping industry, with Stock A (STKA) being mentioned frequently as a stable performer in a volatile market. The report also mentioned that the transportation sector grew by 2 percent last quarter, a healthy sign. "
                    "The report also contains a detailed analysis of import/export trends and a forecast for the agricultural sector. Legacy automaker Stock D (STKD) was mentioned as facing significant challenges with its transition to electric vehicles, with one analyst noting 'serious concerns about their future market share.'\n\n"
                    "**PUBLIC SERVICE ANNOUNCEMENT:**\n"
                    "The Department of Parks and Recreation has reminded citizens not to feed the wildlife in city parks. "
                    "A blood drive is being held at the community center this weekend.\n\n"
                    "**FOOTNOTE:**\n"
                    "A new high-speed railway expansion project will exclusively use Stock B (STKB) for all its signaling equipment. This is projected to boost its price to $65. "
                    "Stock C (STKC), a furniture manufacturer, is expected to remain stable. "
                    "Stock E (STKE) and Stock F (STKF) are expected to have stable pricing."
                )
            },
            # Scenario 14: Capital Cannot Be Fully Deployed
            {
                "name": "Capital Cannot Be Fully Deployed",
                "description": "The best asset is too expensive to use all the available capital.",
                "difficulty": "medium",
                "capital": 1000.0,
                "transaction_cost_per_trade": 20.0,
                "assets": [
                    Asset(id="STKA", name="Stock A", current_price=900.0, expected_price=1000.0),
                    Asset(id="STKB", name="Stock B", current_price=30.0, expected_price=30.0),
                    Asset(id="STKC", name="Stock C", current_price=45.0, expected_price=45.0),
                    Asset(id="STKD", name="Stock D", current_price=150.0, expected_price=140.0),
                    Asset(id="STKE", name="Stock E", current_price=180.0, expected_price=180.0),
                    Asset(id="STKF", name="Stock F", current_price=220.0, expected_price=220.0),
                ],
                "news": (
                    "**URBAN DEVELOPMENT NEWS:**\n"
                    "A new skyscraper, the 'N-Tower', is being built downtown in City N and is set to become the tallest building in the city. The project has been controversial due to its scale. "
                    "The city's historical society is petitioning to have an old theater declared a landmark.\n\n"
                    "**LOCAL EVENTS:**\n"
                    "The annual street art festival will take place this weekend, featuring work from over 50 artists. "
                    "A farmer's market is now open every Sunday in the town square.\n\n"
                    "**RETAIL SECTOR ANALYSIS:**\n"
                    "A new report on consumer spending shows a surge in the luxury market. Stock A (STKA), a high-end fashion brand, is expected to climb to $1000 per share. The CEO of a different luxury company just announced they are launching a new fragrance line. "
                    "The budget retail sector, including consumer electronics seller Stock B (STKB), remains unchanged amidst the focus on high-end goods. "
                    "Stock C (STKC), a pet supply store, is expected to have a stable quarter. "
                    "However, department store Stock D (STKD) is struggling with online competition and analysts have a 'negative outlook' on its future. "
                    "Stock E (STKE) and Stock F (STKF), both specialty retailers, are expected to be stable."
                )
            },
            # Scenario 15: Two Good Choices
            {
                "name": "Two Good Choices",
                "description": "Two different assets offer the exact same ROI, so either is a correct answer.",
                "difficulty": "hard",
                "capital": 50000.0,
                "transaction_cost_per_trade": 50.0,
                "assets": [
                    Asset(id="STKA", name="Stock A", current_price=100.0, expected_price=120.0),
                    Asset(id="STKB", name="Stock B", current_price=50.0, expected_price=60.0),
                    Asset(id="STKC", name="Stock C", current_price=25.0, expected_price=25.0),
                    Asset(id="STKD", name="Stock D", current_price=200.0, expected_price=190.0),
                    Asset(id="STKE", name="Stock E", current_price=300.0, expected_price=300.0),
                    Asset(id="STKF", name="Stock F", current_price=400.0, expected_price=400.0),
                ],
                "news": (
                    "**ACADEMIC CORNER:**\n"
                    "A research paper from the National University's business school in Country O has identified two companies as prime investment opportunities. "
                    "The paper highlights that Stock A (STKA) is predicted to jump to $120 following a major acquisition, and Stock B (STKB) is expected to rise to $60 due to expansion into a new market. "
                    "Both are praised for their strong management and have identical growth prospects. The paper also discusses the importance of diversification and notes that the overall market is up 1 percent this month. "
                    "It also mentioned that a former high-flyer, Stock D (STKD), is 'facing significant headwinds' due to outdated technology.\n\n"
                    "**COMMUNITY BULLETIN:**\n"
                    "The annual town fair will be held next month and will feature a pie-eating contest and a Ferris wheel. "
                    "The public library is holding a book drive for local schools.\n\n"
                    "**MEDIA MAVEN:**\n"
                    "A popular morning show host has announced their retirement after 25 years on the air. "
                    "A new documentary about the history of video games is trending on a major streaming service. "
                    "Stock C (STKC) is a company that produces office supplies, and its outlook is flat for the coming quarter. "
                    "Stock E (STKE) and Stock F (STKF) are not expected to see any price changes."
                )
            }
        ]

    def _create_prompt(self, scenario: Dict[str, Any]) -> str:
        """Create a prompt for the given portfolio optimization scenario."""
        asset_list = scenario['assets']
        asset_tickers = ", ".join([f"{asset.id}: 0" for asset in asset_list])
        
        prompt = (
            "You are a financial analyst and portfolio manager. Your goal is to allocate your available "
            f"capital of ${scenario['capital']:,.2f} to maximize the expected profit for the **single upcoming time period**. "
            "Your decisions should be based only on achieving the best **short-term** outcome.\n\n"
            "Analyze the following news report to determine the best investment strategy. Your decision should "
            "be based *only* on the information provided here.\n\n"
            "--- NEWS REPORT ---\n"
            f"{scenario['news']}\n"
            "--- END REPORT ---\n\n"
            "AVAILABLE ASSETS:\n"
        )
        for asset in asset_list:
            prompt += f"- {asset.name} (Ticker: {asset.id}), Current Price: ${asset.current_price:.2f}\n"

        prompt += (
            f"\nCONSTRAINTS & COSTS:\n"
            "- You must allocate your capital to at most one of the available assets.\n"
            f"- **Transaction Cost**: Each trade you make incurs a single, fixed fee of ${scenario['transaction_cost_per_trade']:.2f}. "
            "This is a one-time cost per asset bought, not per share. For example, buying 100 shares of STKA costs "
            f"(100 * price of STKA) + ${scenario['transaction_cost_per_trade']:.2f}.\n"
            "- **Capital Limit**: You cannot spend more than your total available capital, including all share costs and transaction fees.\n"
            "- **Whole Shares**: Partial shares are not allowed; you must buy whole numbers of shares.\n\n"
            "After your analysis, provide your final answer in a specific format on a single line. "
            "You must include all available asset tickers. For assets you do not wish to buy, list their shares as 0.\n"
            "Format your answer exactly like this: [FINAL ANSWER: Ticker1: Shares, Ticker2: Shares, ...]\n\n"
            f"Example for this scenario: [FINAL ANSWER: {asset_tickers}]"
        )
        return prompt

    def _parse_response(
        self, response_text: str, assets: List[Asset]
    ) -> Portfolio:
        """
        Parses the model's text response to extract the portfolio trades.
        It respects the last-mentioned allocation for each ticker. It first
        looks for the specific [FINAL ANSWER: ...] format. If that is not found
        or yields no valid trades, it falls back to parsing the entire response.
        """
        valid_asset_ids = {asset.id for asset in assets}
        trades_map = {}

        def parse_block(text_block: str) -> Dict[str, int]:
            """Finds all 'TICKER: shares' pairs and returns a map, respecting the last one seen."""
            found_trades = {}
            pattern = r"\b([A-Z]+):\s*(\d+)\b"
            matches = re.findall(pattern, text_block, re.IGNORECASE)
            for asset_id, shares_str in matches:
                if asset_id.upper() in valid_asset_ids:
                    found_trades[asset_id.upper()] = int(shares_str)
            return found_trades

        # Primary strategy: parse the [FINAL ANSWER: ...] block
        final_answer_match = re.search(r"\[FINAL ANSWER:\s*(.*?)\]", response_text, re.IGNORECASE | re.DOTALL)
        if final_answer_match:
            answer_block = final_answer_match.group(1)
            trades_map = parse_block(answer_block)

        # Fallback strategy: if the primary strategy fails, parse the whole response
        if not trades_map:
            trades_map = parse_block(response_text)

        final_trades = [
            Trade(asset_id=asset_id, shares=shares)
            for asset_id, shares in trades_map.items() if shares > 0
        ]
        
        return Portfolio(trades=final_trades)

    def _calculate_utility(
        self,
        portfolio: Portfolio,
        assets: List[Asset],
        capital: float,
        transaction_cost: float
    ) -> Tuple[float, bool]:
        """
        Calculates the expected utility (profit) of a given portfolio and validates it.
        """
        asset_map = {asset.id: asset for asset in assets}
        total_cost = 0
        expected_payout = 0

        if not portfolio.trades:
            return 0.0, True # Holding is a valid, zero-utility option

        # Explicitly check for violation of the 'at most one asset' rule.
        if len(portfolio.trades) > 1:
            logger.warning(f"Portfolio violates the 'at most one asset' constraint.")
            return 0.0, False

        # Validate trades and calculate total cost
        for trade in portfolio.trades:
            if trade.asset_id not in asset_map:
                logger.warning(f"Invalid asset ID '{trade.asset_id}' in response.")
                return 0.0, False # Invalid asset ID
            if trade.shares <= 0:
                return 0.0, False # Must buy positive shares

            asset = asset_map[trade.asset_id]
            total_cost += (trade.shares * asset.current_price) + transaction_cost

        if total_cost > capital:
            logger.warning(f"Portfolio cost ({total_cost}) exceeds capital ({capital}).")
            return 0.0, False # Exceeded capital

        # If valid, calculate expected utility
        for trade in portfolio.trades:
            asset = asset_map[trade.asset_id]
            # Profit from this trade = (expected_price - current_price) * shares
            expected_payout += (asset.expected_price - asset.current_price) * trade.shares
        
        total_transaction_cost = len(portfolio.trades) * transaction_cost
        net_utility = expected_payout - total_transaction_cost

        return net_utility, True

    def _solve_portfolio(
        self,
        assets: List[Asset],
        capital: float,
        transaction_cost: float
    ) -> Tuple[Portfolio, float]:
        """
        Finds an optimal portfolio by evaluating the total potential utility of
        investing the maximum possible amount in each single profitable asset and
        choosing the asset that yields the highest total profit. This avoids the
        flaw of a simple ROI sort when transaction costs are significant.
        """
        best_portfolio = Portfolio(trades=[])
        max_utility = 0.0

        # Evaluate the potential of investing everything into each single asset
        for asset in assets:
            profit_per_share = asset.expected_price - asset.current_price

            # Only consider assets that are profitable on a per-share basis
            if profit_per_share > 0:
                # Check if we can even afford the transaction cost
                if capital < transaction_cost:
                    continue

                # How many shares can we buy?
                capital_for_shares = capital - transaction_cost
                num_shares = int(capital_for_shares / asset.current_price)

                if num_shares > 0:
                    # Calculate the total profit for this potential trade
                    total_profit = (profit_per_share * num_shares) - transaction_cost

                    # If this single-asset portfolio is better than what we've found so far
                    if total_profit > max_utility:
                        max_utility = total_profit
                        best_portfolio = Portfolio(trades=[Trade(asset_id=asset.id, shares=num_shares)])

        return best_portfolio, max_utility


    async def evaluate_task(self, task: Task, model: BaseModel) -> TaskResult:
        """Evaluate a single portfolio optimization task."""
        start_time = time.time()
        
        scenario = task.metadata['scenario']
        assets = [Asset(**asdict(a)) for a in scenario['assets']]
        capital = scenario['capital']
        transaction_cost = scenario['transaction_cost_per_trade']
        
        model_response = await model.generate(task.prompt)
        execution_time = float(model_response.latency or 0.0)

        parsed_portfolio = self._parse_response(model_response.text, assets)

        achieved_utility, is_valid = self._calculate_utility(
            parsed_portfolio, assets, capital, transaction_cost
        )
        optimal_utility = task.metadata['optimal_utility']
        
        score = 0
        if optimal_utility > 0 and is_valid:
            score = achieved_utility / optimal_utility
        elif optimal_utility == 0 and achieved_utility == 0 and is_valid:
            score = 1.0

        return TaskResult(
            task_id=task.task_id,
            task_name=task.name,
            agent_type=self.agent_type,
            success=is_valid and score >= 0.95, # Success if score is at least 95% of optimal
            score=max(0, min(score, 1.0)), # Clamp score between 0 and 1
            metrics={
                "achieved_utility": achieved_utility,
                "optimal_utility": optimal_utility,
                "is_valid_portfolio": is_valid,
                "optimality_gap": optimal_utility - achieved_utility,
                "output_tokens": model_response.completion_tokens,
            },
            model_response=model_response,
            execution_time=execution_time,
        )
    
    def calculate_score(self, task: Task, model_response: ModelResponse) -> float:
        # Handled in evaluate_task
        return 0.0
