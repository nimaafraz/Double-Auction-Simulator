#!/usr/bin/env python3
"""
Auction Simulator for Sealed-Bid Multi-item Double Auction
-----------------------------------------------------------------------------------------------
This script simulates an auction market where agents participate as buyers and sellers.
Commodity units (frames) are treated as integers.

Auction mechanisms implemented include:
  • Nima-McAfee Auction
  • MUDA Auction
  • SBBA Auction (Strongly Budget-Balanced, simplified)
  • VCG-Based Auction (simplified)
  • Upper Bound (benchmark)

The final detailed CSV file includes the common columns:
  Round, agent ID, Type, bid/ask, unit demand
and for each mechanism the outcome columns (with mechanism-specific suffix):
  Number of traded units, price of trade, agent utility, total social welfare of all agents

Social welfare is computed in three ways:
  SW_GFT   = ∑ (buyer bid - seller ask) × quantity traded.
  SW_Util  = Sum of individual agent utilities.
  SW_Raw   = Minimum agent utility among all agents.

Author: Nima Afraz, University College Dublin
Date: 2025-03-13
License: MIT License
"""

import time
import random
import numpy as np
import pandas as pd
import copy
from collections import defaultdict

# ---------------------------
# Agent Class Definition
# ---------------------------
class Agent:
    """
    Represents an agent participating in the auction.
    
    Attributes:
        id (int): Unique identifier for the agent.
        valuation (float): For buyers, this is the bid; for sellers, this is the ask.
        bid_type (str): 'buy' or 'sell'.
        quantity (int): Number of commodity units (frames).
    """
    def __init__(self, id, valuation, bid_type='buy', quantity=1):
        self.id = id
        self.valuation = valuation
        self.bid_type = bid_type
        self.quantity = quantity  # integer commodity units

    def __repr__(self):
        return (f"Agent(id={self.id}, valuation={self.valuation}, "
                f"bid_type='{self.bid_type}', quantity={self.quantity})")

# ---------------------------
# Helper Functions
# ---------------------------
def expand_units(participants):
    """
    Expands each Agent into a list of unit offers: (id, valuation).
    """
    units = []
    for p in participants:
        for _ in range(p.quantity):
            units.append((p.id, p.valuation))
    return units

# ---------------------------
# Social Welfare Measures
# ---------------------------
def compute_sw_gft(transactions, buyers, sellers):
    """
    Gain-from-trade social welfare:
      SW_GFT = ∑ (buyer bid - seller ask) × quantity traded.
    """
    sw = 0.0
    for (b_id, s_id, qty, price) in transactions:
        buyer_val = next(b.valuation for b in buyers if b.id == b_id)
        seller_val = next(s.valuation for s in sellers if s.id == s_id)
        sw += (buyer_val - seller_val) * qty
    return sw

def compute_sw_util(df):
    """
    Utilitarian social welfare:
      SW_Util = Sum of individual agent utilities.
    """
    return df["agent utility"].sum()

def compute_sw_raw(df):
    """
    Rawlsian social welfare:
      SW_Raw = Minimum agent utility among all agents.
    """
    return df["agent utility"].min() if not df.empty else 0

def calculate_upper_bound(buyers, sellers):
    """
    Computes the maximum possible social welfare by matching unit offers.
    """
    buyer_units = expand_units(buyers)
    seller_units = expand_units(sellers)
    buyer_units.sort(key=lambda x: x[1], reverse=True)
    seller_units.sort(key=lambda x: x[1])
    ub = 0.0
    max_units = min(len(buyer_units), len(seller_units))
    for i in range(max_units):
        if buyer_units[i][1] >= seller_units[i][1]:
            ub += buyer_units[i][1] - seller_units[i][1]
        else:
            break
    return ub

# ---------------------------
# Auction Mechanism Implementations
# ---------------------------

def aggregated_matching(buyers, sellers, mechanism="basic"):
    """
    Basic aggregated matching:
      - Sort buyers (desc) and sellers (asc).
      - Each match trades min(buyer.quantity, seller.quantity).
      - Price is the average of buyer bid and seller ask.
    Returns:
        List of transactions: (buyer_id, seller_id, quantity, price)
    """
    buyers_sorted = sorted(copy.deepcopy(buyers), key=lambda b: b.valuation, reverse=True)
    sellers_sorted = sorted(copy.deepcopy(sellers), key=lambda s: s.valuation)
    transactions = []
    i = j = 0
    while i < len(buyers_sorted) and j < len(sellers_sorted):
        if buyers_sorted[i].valuation >= sellers_sorted[j].valuation:
            qty = min(buyers_sorted[i].quantity, sellers_sorted[j].quantity)
            price = (buyers_sorted[i].valuation + sellers_sorted[j].valuation) / 2.0
            transactions.append((buyers_sorted[i].id, sellers_sorted[j].id, qty, price))
            buyers_sorted[i].quantity -= qty
            sellers_sorted[j].quantity -= qty
            if buyers_sorted[i].quantity == 0:
                i += 1
            if sellers_sorted[j].quantity == 0:
                j += 1
        else:
            break
    return transactions

def aggregated_nima_mcafee(buyers, sellers):
    """
    Aggregated Nima-McAfee Auction:
      - Sort buyers (desc) and sellers (asc).
      - Let k be the largest index such that for all i ≤ k, buyer[i].valuation ≥ seller[i].valuation.
      - Compute uniform price using the (k+1)th pair if available.
      - Execute trades for the first k-1 pairs.
    Returns:
        (transactions, uniform_price)
    """
    buyers_sorted = sorted(copy.deepcopy(buyers), key=lambda b: b.valuation, reverse=True)
    sellers_sorted = sorted(copy.deepcopy(sellers), key=lambda s: s.valuation)
    k = 0
    max_pairs = min(len(buyers_sorted), len(sellers_sorted))
    while k < max_pairs and buyers_sorted[k].valuation >= sellers_sorted[k].valuation:
        k += 1
    if k <= 1:
        return [], 0
    if k < max_pairs:
        uniform_price = (buyers_sorted[k].valuation + sellers_sorted[k].valuation) / 2.0
    else:
        uniform_price = (buyers_sorted[k-1].valuation + sellers_sorted[k-1].valuation) / 2.0
    transactions = []
    for i in range(k-1):
        qty = min(buyers_sorted[i].quantity, sellers_sorted[i].quantity)
        transactions.append((buyers_sorted[i].id, sellers_sorted[i].id, qty, uniform_price))
        buyers_sorted[i].quantity -= qty
        sellers_sorted[i].quantity -= qty
    return transactions, uniform_price

def muda_auction(buyers, sellers):
    """
    Simplified MUDA Auction.
      1. Randomly partition buyers into B1 and B2; sellers into S1 and S2.
      2. Compute clearing prices p1 and p2 via aggregated matching on (B1,S1) and (B2,S2).
      3. Cross-match: Buyers in B1 with sellers in S2 at p2; Buyers in B2 with sellers in S1 at p1.
    Returns:
        List of transactions: (buyer_id, seller_id, quantity, price)
    """
    B1, B2 = [], []
    for b in buyers:
        (B1 if random.random() < 0.5 else B2).append(copy.deepcopy(b))
    S1, S2 = [], []
    for s in sellers:
        (S1 if random.random() < 0.5 else S2).append(copy.deepcopy(s))
    trades_g1 = aggregated_matching(B1, S1)
    trades_g2 = aggregated_matching(B2, S2)
    p1 = sum(price for (_, _, _, price) in trades_g1) / len(trades_g1) if trades_g1 else 0
    p2 = sum(price for (_, _, _, price) in trades_g2) / len(trades_g2) if trades_g2 else 0
    transactions = []
    for b in B1:
        for s in S2:
            if b.valuation >= p2 and s.valuation <= p2:
                qty = min(b.quantity, s.quantity)
                if qty > 0:
                    transactions.append((b.id, s.id, qty, p2))
                    b.quantity -= qty
                    s.quantity -= qty
    for b in B2:
        for s in S1:
            if b.valuation >= p1 and s.valuation <= p1:
                qty = min(b.quantity, s.quantity)
                if qty > 0:
                    transactions.append((b.id, s.id, qty, p1))
                    b.quantity -= qty
                    s.quantity -= qty
    return transactions

def sbba_auction(buyers, sellers):
    """
    Simplified SBBA Auction.
      1. Sort buyers (desc) and sellers (asc).
      2. Let k be the largest index such that for all i ≤ k, buyer[i].valuation ≥ seller[i].valuation.
      3. Randomly drop one seller from the first k sellers to ensure strong budget balance.
      4. Compute uniform price p using an index that does not exceed the remaining sellers.
      5. Execute trades for the remaining k-1 pairs at price p.
    Returns:
        List of transactions: (buyer_id, seller_id, quantity, price)
    """
    buyers_sorted = sorted(copy.deepcopy(buyers), key=lambda b: b.valuation, reverse=True)
    sellers_sorted = sorted(copy.deepcopy(sellers), key=lambda s: s.valuation)
    k = 0
    max_pairs = min(len(buyers_sorted), len(sellers_sorted))
    while k < max_pairs and buyers_sorted[k].valuation >= sellers_sorted[k].valuation:
        k += 1
    if k <= 1:
        return []
    drop_index = random.randint(0, k-1)
    sellers_sorted.pop(drop_index)  # Drop one seller
    effective_k = k - 1
    idx = effective_k if effective_k < len(sellers_sorted) else effective_k - 1
    p = (buyers_sorted[idx].valuation + sellers_sorted[idx].valuation) / 2.0
    transactions = []
    for i in range(effective_k):
        qty = min(buyers_sorted[i].quantity, sellers_sorted[i].quantity)
        transactions.append((buyers_sorted[i].id, sellers_sorted[i].id, qty, p))
        buyers_sorted[i].quantity -= qty
        sellers_sorted[i].quantity -= qty
    return transactions

def vcg_auction(buyers, sellers):
    """
    Simplified VCG-Based Auction for multi-unit double auctions.
      - Compute allocation via greedy matching.
      - Let k be the number of matched pairs.
      - Determine critical prices: for buyers, the (k+1)th bid (if exists); for sellers, the (k+1)th ask.
      - Set uniform price p = (critical_bid + critical_ask) / 2.
      - Execute trades for the first k-1 pairs at price p.
    Returns:
        List of transactions: (buyer_id, seller_id, quantity, price)
    """
    buyers_sorted = sorted(copy.deepcopy(buyers), key=lambda b: b.valuation, reverse=True)
    sellers_sorted = sorted(copy.deepcopy(sellers), key=lambda s: s.valuation)
    k = 0
    max_pairs = min(len(buyers_sorted), len(sellers_sorted))
    while k < max_pairs and buyers_sorted[k].valuation >= sellers_sorted[k].valuation:
        k += 1
    if k <= 1:
        return []
    if k < len(buyers_sorted):
        critical_bid = buyers_sorted[k].valuation
    else:
        critical_bid = buyers_sorted[k-1].valuation
    if k < len(sellers_sorted):
        critical_ask = sellers_sorted[k].valuation
    else:
        critical_ask = sellers_sorted[k-1].valuation
    p = (critical_bid + critical_ask) / 2.0
    transactions = []
    for i in range(k-1):
        qty = min(buyers_sorted[i].quantity, sellers_sorted[i].quantity)
        transactions.append((buyers_sorted[i].id, sellers_sorted[i].id, qty, p))
        buyers_sorted[i].quantity -= qty
        sellers_sorted[i].quantity -= qty
    return transactions

# ---------------------------
# Detailed Outcome Table Generator
# ---------------------------
def detailed_outcome_table(buyers, sellers, transactions, mechanism_name, round_num):
    """
    Generates a detailed outcome table (one row per agent) with columns:
      Round, agent ID, Type, bid/ask, unit demand, Number of traded units,
      price of trade, agent utility, total social welfare of all agents, Mechanism.
    
    For buyers, 'bid/ask' is their bid; for sellers, it is their ask.
    'unit demand' is the original commodity units.
    """
    buyer_trade = defaultdict(int)
    seller_trade = defaultdict(int)
    buyer_prices = defaultdict(list)
    seller_prices = defaultdict(list)
    for (b_id, s_id, qty, price) in transactions:
        buyer_trade[b_id] += qty
        seller_trade[s_id] += qty
        buyer_prices[b_id].append(price)
        seller_prices[s_id].append(price)
    rows = []
    for b in buyers:
        traded = buyer_trade[b.id]
        avg_price = round(sum(buyer_prices[b.id]) / len(buyer_prices[b.id]), 3) if buyer_prices[b.id] else 0
        utility = round((b.valuation - avg_price) * traded, 3) if traded > 0 else 0
        rows.append({
            "Round": round_num,
            "agent ID": b.id,
            "Type": "buyer",
            "bid/ask": b.valuation,
            "unit demand": b.quantity,
            "Number of traded units": traded,
            "price of trade": avg_price,
            "agent utility": utility
        })
    for s in sellers:
        traded = seller_trade[s.id]
        avg_price = round(sum(seller_prices[s.id]) / len(seller_prices[s.id]), 3) if seller_prices[s.id] else 0
        utility = round((avg_price - s.valuation) * traded, 3) if traded > 0 else 0
        rows.append({
            "Round": round_num,
            "agent ID": s.id,
            "Type": "seller",
            "bid/ask": s.valuation,
            "unit demand": s.quantity,
            "Number of traded units": traded,
            "price of trade": avg_price,
            "agent utility": utility
        })
    df = pd.DataFrame(rows, columns=["Round", "agent ID", "Type", "bid/ask", "unit demand",
                                       "Number of traded units", "price of trade", "agent utility"])
    # Ensure the social welfare column is present even if no transactions occurred.
    sw_gft = compute_sw_gft(transactions, buyers, sellers)
    if "total social welfare of all agents" not in df.columns:
        df["total social welfare of all agents"] = sw_gft
    else:
        df["total social welfare of all agents"] = sw_gft
    df["Mechanism"] = mechanism_name
    return df

# ---------------------------
# Outcome Generator per Mechanism for One Market Instance
# ---------------------------
def get_outcome_for_mechanism(mech, buyers, sellers, round_num):
    """
    For a given market instance and mechanism name, computes the detailed outcome table.
      - "Nima-McAfee Auction": uses aggregated_nima_mcafee.
      - "MUDA Auction": uses muda_auction.
      - "SBBA Auction": uses sbba_auction.
      - "VCG-Based Auction": uses vcg_auction.
      - "Upper Bound": uses aggregated_matching (basic) then overrides prices.
    Returns:
        A DataFrame with detailed outcome rows.
    """
    if mech == "Nima-McAfee Auction":
        transactions, _ = aggregated_nima_mcafee(buyers, sellers)
        df = detailed_outcome_table(buyers, sellers, transactions, mech, round_num)
    elif mech == "MUDA Auction":
        transactions = muda_auction(buyers, sellers)
        df = detailed_outcome_table(buyers, sellers, transactions, mech, round_num)
    elif mech == "SBBA Auction":
        transactions = sbba_auction(buyers, sellers)
        df = detailed_outcome_table(buyers, sellers, transactions, mech, round_num)
    elif mech == "VCG-Based Auction":
        transactions = vcg_auction(buyers, sellers)
        df = detailed_outcome_table(buyers, sellers, transactions, mech, round_num)
    elif mech == "Upper Bound":
        transactions = aggregated_matching(buyers, sellers, mechanism="basic")
        df = detailed_outcome_table(buyers, sellers, transactions, mech, round_num)
        for idx, row in df.iterrows():
            if row["Type"] == "buyer":
                df.at[idx, "price of trade"] = row["bid/ask"]
                df.at[idx, "agent utility"] = 0
            else:
                df.at[idx, "price of trade"] = row["bid/ask"]
                df.at[idx, "agent utility"] = 0
    else:
        key = mech.lower().split()[0]
        transactions = aggregated_matching(buyers, sellers, mechanism=key)
        df = detailed_outcome_table(buyers, sellers, transactions, mech, round_num)
    return df

# ---------------------------
# Common Participant Table Generator
# ---------------------------
def get_common_table(buyers, sellers, round_num):
    """
    Generates a common table for the round with one row per agent containing:
      Round, agent ID, Type, bid/ask, unit demand.
    """
    rows = []
    for b in buyers:
        rows.append({
            "Round": round_num,
            "agent ID": b.id,
            "Type": "buyer",
            "bid/ask": b.valuation,
            "unit demand": b.quantity
        })
    for s in sellers:
        rows.append({
            "Round": round_num,
            "agent ID": s.id,
            "Type": "seller",
            "bid/ask": s.valuation,
            "unit demand": s.quantity
        })
    return pd.DataFrame(rows, columns=["Round", "agent ID", "Type", "bid/ask", "unit demand"])

# ---------------------------
# Simulate Comparative Detailed CSV (One Row per Agent per Round)
# ---------------------------
def simulate_comparative_detailed_csv(num_rounds, num_agents, seed_start=42):
    """
    For each round, generates a market instance and computes, for each mechanism,
    the detailed outcome table. Merges the outcome columns (renamed with mechanism-specific suffixes)
    with the common participant table so that each row (per agent) includes outcome columns for every mechanism.
    All rounds are concatenated and saved to "comparative_auction_results.csv".
    """
    mechanisms = [
        "Nima-McAfee Auction",
        "MUDA Auction",
        "SBBA Auction",
        "VCG-Based Auction",
        "Upper Bound"
    ]
    suffix_map = {
        "Nima-McAfee Auction": "NimaMcAfee",
        "MUDA Auction": "MUDA",
        "SBBA Auction": "SBBA",
        "VCG-Based Auction": "VCG",
        "Upper Bound": "UpperBound"
    }
    all_rounds = []
    for r in range(num_rounds):
        seed = seed_start + r
        random.seed(seed)
        np.random.seed(seed)
        total = num_agents * 2
        market = []
        for i in range(total):
            if i < num_agents:
                valuation = round(random.uniform(0.5, 1.0), 3)
                bid_type = 'buy'
                quantity = random.randint(200, 1000)  # integer commodity units
            else:
                valuation = round(random.uniform(0, 0.5), 3)
                bid_type = 'sell'
                quantity = random.randint(400, 1000)  # integer commodity units
            market.append(Agent(i, valuation, bid_type, quantity))
        buyers = [v for v in market if v.bid_type == 'buy']
        sellers = [v for v in market if v.bid_type == 'sell']
        common_df = get_common_table(buyers, sellers, r+1)
        mech_dfs = []
        for mech in mechanisms:
            df_mech = get_outcome_for_mechanism(mech, copy.deepcopy(buyers), copy.deepcopy(sellers), r+1)
            outcome_cols = ["Number of traded units", "price of trade", "agent utility", "total social welfare of all agents"]
            df_mech_subset = df_mech[["Round", "agent ID", "Type"] + outcome_cols].copy()
            suffix = "_" + suffix_map[mech]
            rename_dict = {
                "Number of traded units": "Number of traded units" + suffix,
                "price of trade": "price of trade" + suffix,
                "agent utility": "agent utility" + suffix,
                "total social welfare of all agents": "total social welfare of all agents" + suffix
            }
            df_mech_subset = df_mech_subset.rename(columns=rename_dict)
            mech_dfs.append(df_mech_subset)
        merged_df = common_df.copy()
        for df_mech_subset in mech_dfs:
            merged_df = pd.merge(merged_df, df_mech_subset, on=["Round", "agent ID", "Type"], how="left")
        all_rounds.append(merged_df)
    full_df = pd.concat(all_rounds, ignore_index=True)
    full_df.to_csv("comparative_auction_results.csv", index=False)
    print("Comparative detailed results saved to comparative_auction_results.csv")
    return full_df

# ---------------------------
# Average Metrics Simulation (Comparative Summary)
# ---------------------------
def simulate_average_metrics(num_rounds, num_agents, seed_start=42):
    """
    For each auction mechanism (excluding Upper Bound, the benchmark),
    runs num_rounds simulations using the same market generation process.
    For each round and mechanism, computes:
      - Social Welfare (GFT): ∑ (buyer bid - seller ask) × quantity traded.
      - Social Welfare (Utilitarian): sum of individual agent utilities.
      - Social Welfare (Rawlsian): minimum agent utility.
      - Transactions, Runtime.
    The summary table shows averages across rounds.
    The 'Rounds' column shows the total number of rounds simulated.
    Returns:
        A summary DataFrame.
    """
    mechanisms = {
        "Nima-McAfee Auction": aggregated_nima_mcafee,
        "MUDA Auction": muda_auction,
        "SBBA Auction": sbba_auction,
        "VCG-Based Auction": vcg_auction
    }
    results = []
    total_round_ub = 0.0
    for r in range(num_rounds):
        seed = seed_start + r
        random.seed(seed)
        np.random.seed(seed)
        total = num_agents * 2
        market = []
        for i in range(total):
            if i < num_agents:
                valuation = round(random.uniform(0.5, 1.0), 2)
                bid_type = 'buy'
                quantity = random.randint(200, 1000)
            else:
                valuation = round(random.uniform(0, 0.5), 2)
                bid_type = 'sell'
                quantity = random.randint(400, 1000)
            market.append(Agent(i, valuation, bid_type, quantity))
        ub = calculate_upper_bound([v for v in market if v.bid_type == 'buy'],
                                    [v for v in market if v.bid_type == 'sell'])
        total_round_ub += ub
        for mech_name, func in mechanisms.items():
            market_copy = copy.deepcopy(market)
            buyers = [v for v in market_copy if v.bid_type == 'buy']
            sellers = [v for v in market_copy if v.bid_type == 'sell']
            start_time = time.time()
            if mech_name == "Nima-McAfee Auction":
                transactions, _ = func(buyers, sellers)
            elif mech_name in ["MUDA Auction"]:
                transactions = func(buyers, sellers)
            else:
                transactions = func(buyers, sellers)
            elapsed = time.time() - start_time
            sw_gft = compute_sw_gft(transactions, buyers, sellers)
            # Use detailed outcome table to compute individual utilities:
            df_outcome = detailed_outcome_table(buyers, sellers, transactions, mech_name, r+1)
            sw_util = compute_sw_util(df_outcome)
            sw_raw = compute_sw_raw(df_outcome)
            tx = len(transactions)
            results.append({
                "Mechanism": mech_name,
                "Rounds": num_rounds,
                "Social Welfare (GFT)": sw_gft,
                "Social Welfare (Utilitarian)": sw_util,
                "Social Welfare (Rawlsian)": sw_raw,
                "Transactions": tx,
                "Runtime (s)": elapsed,
                "Upper Bound": ub
            })
    df_all = pd.DataFrame(results)
    df_summary = df_all.groupby("Mechanism").mean().reset_index()
    df_summary["Rounds"] = num_rounds  # Set Rounds to total rounds simulated
    avg_ub = round(total_round_ub / num_rounds, 2)
    df_summary["Upper Bound"] = avg_ub
    overall_row = pd.DataFrame([{
        "Mechanism": "Upper Bound",
        "Rounds": num_rounds,
        "Social Welfare (GFT)": avg_ub,
        "Social Welfare (Utilitarian)": avg_ub,
        "Social Welfare (Rawlsian)": avg_ub,
        "Transactions": "-",
        "Runtime (s)": "-",
        "Upper Bound": avg_ub
    }])
    df_summary = pd.concat([df_summary, overall_row], ignore_index=True)
    return df_summary

# ---------------------------
# Main Execution
# ---------------------------
def main():
    try:
        num_rounds = int(input("Enter number of auction rounds to simulate: "))
    except:
        num_rounds = 100
        print("Invalid input. Using default of 100 rounds.")
    try:
        num_agents = int(input("Enter number of buyers (and equal number of sellers): "))
    except:
        num_agents = 20
        print("Invalid input. Using default of 20.")
    
    print("\nSimulating average metrics for each mechanism...")
    df_summary = simulate_average_metrics(num_rounds, num_agents)
    print("\nAverage Metrics Comparison:")
    print(df_summary.to_string(index=False))
    df_summary.to_csv("auction_simulation_results.csv", index=False)
    print("\nAverage metrics results saved to auction_simulation_results.csv")
    
    print("\nSimulating detailed comparative outcomes for each round and mechanism...")
    df_comparative = simulate_comparative_detailed_csv(num_rounds, num_agents)
    print(df_comparative.head())

if __name__ == '__main__':
    main()
