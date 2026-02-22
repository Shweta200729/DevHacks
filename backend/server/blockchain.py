import os
import logging
from web3 import Web3
from eth_tester import EthereumTester

logger = logging.getLogger(__name__)

# Initialize Ethereum Tester and Web3
tester = EthereumTester()
w3 = Web3(Web3.EthereumTesterProvider(tester))

STAKING_POOL_ADDRESS = w3.eth.accounts[0]

# State maps
client_wallets = {}
staked_balances = {}
transaction_log = []

def get_or_create_wallet(client_id: str) -> str:
    if client_id in client_wallets:
        return client_wallets[client_id]
    
    num_assigned = len(client_wallets) + 1 # +1 because 0 is pool
    if num_assigned < len(w3.eth.accounts):
        wallet = w3.eth.accounts[num_assigned]
    else:
        # Fallback to rotating accounts if we exceed 9 clients
        wallet = w3.eth.accounts[(num_assigned % 9) + 1]
        
    client_wallets[client_id] = wallet
    
    # Automatically stake 100 FLT upon registration
    stake_tokens(client_id, 100)
    return wallet

def log_tx(tx_hash, action, client_id, amount):
    try:
        receipt = w3.eth.get_transaction_receipt(tx_hash)
        transaction_log.append({
            "tx_hash": tx_hash.hex(),
            "action": action,
            "client_id": client_id,
            "amount": amount,
            "block_number": receipt.blockNumber,
            "gas_used": receipt.gasUsed
        })
        # Keep only last 50 transactions
        if len(transaction_log) > 50:
            transaction_log.pop(0)
    except Exception as e:
        logger.error(f"Failed to log transaction: {e}")

def stake_tokens(client_id: str, amount: int = 100):
    wallet = client_wallets.get(client_id)
    if not wallet: return None
    tx_hash = w3.eth.send_transaction({
        "from": wallet,
        "to": STAKING_POOL_ADDRESS,
        "value": w3.to_wei(amount, "ether"),
        "gas": 21000
    })
    staked_balances[wallet] = staked_balances.get(wallet, 0) + amount
    log_tx(tx_hash, "STAKE", client_id, amount)
    return tx_hash.hex()

def reward_client(client_id: str, amount: int = 10):
    wallet = client_wallets.get(client_id)
    if not wallet: return None
    tx_hash = w3.eth.send_transaction({
        "from": STAKING_POOL_ADDRESS,
        "to": wallet,
        "value": w3.to_wei(amount, "ether"),
        "gas": 21000
    })
    log_tx(tx_hash, "REWARD", client_id, amount)
    return tx_hash.hex()

def slash_client(client_id: str, amount: int = 15):
    wallet = client_wallets.get(client_id)
    if not wallet: return None
    
    slash_amount = min(amount, staked_balances.get(wallet, 0))
    staked_balances[wallet] -= slash_amount
    
    # Simulate a slashing transaction permanently on chain
    tx_hash = w3.eth.send_transaction({
        "from": STAKING_POOL_ADDRESS,
        "to": wallet,
        "value": 0,
        "data": b"SLASH",
        "gas": 30000
    })
    log_tx(tx_hash, "SLASH", client_id, slash_amount)
    return tx_hash.hex()

def get_client_status(client_id: str):
    wallet = client_wallets.get(client_id)
    if not wallet: return None
    bal_wei = w3.eth.get_balance(wallet)
    return {
        "wallet": wallet,
        "balance": float(w3.from_wei(bal_wei, "ether")),
        "staked": staked_balances.get(wallet, 0)
    }

def get_all_status():
    res = []
    for cid, w in client_wallets.items():
        res.append({
            "client_id": cid,
            "wallet": w,
            "balance": round(float(w3.from_wei(w3.eth.get_balance(w), "ether")), 2),
            "staked": staked_balances.get(w, 0)
        })
    return res

def get_recent_transactions():
    return transaction_log[::-1]
