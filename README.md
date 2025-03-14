
# Auction Simulator for Sealed-Bid Multi-item Double Auction

## Overview
This repository contains a Python-based auction simulator designed to evaluate and compare different sealed-bid, multi-item double auction mechanisms.

## Auction Mechanisms Implemented
- **Nima-McAfee Auction** [Afraz & Ruffini, 2018](https://doi.org/10.1109/JLT.2018.2875188)
- **MUDA Auction** [[paper](https://arxiv.org/abs/1712.06848)]
- **SBBA Auction** (Strongly Budget-Balanced Auction) [[paper](https://doi.org/10.1007/978-3-662-53354-3_21)]
- **VCG-Based Auction** (Vickrey-Clarke-Groves)
- **Upper Bound** (benchmark optimal allocation)

## How to Run
Ensure Python 3.6+ is installed, and install dependencies with:
```bash
pip install numpy pandas
```

Then, run the script via command line:
```bash
python test.py
```

Follow the prompts to enter the number of auction rounds and agents.

## Outputs
Two CSV files are generated:
- `comparative_auction_results.csv`: Detailed outcomes for each agent per round.
- `auction_simulation_results.csv`: Average summary metrics across auction rounds.

## CSV File Format
The detailed CSV includes:

**Common columns:**
- `Round`
- `agent ID`
- `Type` (buyer/seller)
- `bid/ask`
- `unit demand`

**For each mechanism** (with suffixes `_NimaMcAfee`, `_MUDA`, `_SBBA`, `_VCG`, `_UpperBound`):
- `Number of traded units`
- `price of trade`
- `agent utility`
- `total social welfare of all agents`

## Social Welfare Definitions
- **Gain-from-Trade (SW_GFT)**: Sum of differences between buyers' valuations and sellers' asks.
- **Utilitarian Welfare (SW_Util)**: Sum of utilities for all participants.
- **Rawlsian Welfare (SW_Raw)**: Minimum utility achieved by any single participant.

## Requirements
- Python ≥3.6
- pandas, numpy

Install dependencies with:
```bash
pip install numpy pandas
```

## References
- Segal-Halevi, E., Hassidim, A., & Aumann, Y. (2017). MUDA: A Truthful Multi-Unit Double-Auction Mechanism. [arXiv:1712.06848](https://arxiv.org/abs/1712.06848).
- Segal-Halevi, E., Hassidim, A., & Aumann, Y. (2016). SBBA: A Strongly-Budget-Balanced Double-Auction Mechanism. In *Algorithmic Game Theory*, Springer. [DOI:10.1007/978-3-662-53354-3_21](https://doi.org/10.1007/978-3-662-53354-3_21)
- Afraz, N., & Ruffini, M. (2018). A Sharing Platform for Multi-Tenant PONs. *Journal of Lightwave Technology*, 36(23), 5413-5423. [DOI:10.1109/JLT.2018.2875188](https://doi.org/10.1109/JLT.2018.2875188)

## Author
**Nima Afraz**, University College Dublin

## License
MIT License
