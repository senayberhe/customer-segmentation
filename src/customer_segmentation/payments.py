"""Payment processing module for customer segmentation pipeline."""

from abc import ABC, abstractmethod
from datetime import datetime


class PaymentProcessor(ABC):
    def __init__(self, api_key: str, currency: str = "USD"):
        self.api_key = api_key
        self.currency = currency
        self._log: list[str] = []

    def log(self, msg: str) -> None:
        entry = f"[{datetime.now():%H:%M:%S}] {msg}"
        self._log.append(entry)
        print(entry)

    def charge(self, amount: float, description: str) -> dict:
        self.log(f"Charging {self.currency} {amount:.2f} - {description}")
        result = self._do_charge(amount, description)
        self.log(f"Result: {result['status']}")
        return result

    @abstractmethod
    def _do_charge(self, amount: float, description: str) -> dict: ...

    @abstractmethod
    def refund(self, transaction_id: str) -> bool: ...


class StripeProcessor(PaymentProcessor):
    def __init__(self, api_key: str, webhook_secret: str):
        super().__init__(api_key, currency="USD")
        self.webhook_secret = webhook_secret

    def _do_charge(self, amount: float, description: str) -> dict:
        return {
            "status": "succeeded",
            "provider": "stripe",
            "id": "pi_3NK4AB",
            "amount": amount,
        }

    def refund(self, transaction_id: str) -> bool:
        self.log(f"Stripe refund for {transaction_id}")
        return True

    def verify_webhook(self, payload: bytes, sig: str) -> bool:
        return sig == self.webhook_secret


class PayPalProcessor(PaymentProcessor):
    def __init__(self, api_key: str, client_id: str, sandbox: bool = False):
        super().__init__(api_key)
        self.client_id = client_id
        self.sandbox = sandbox
        self._base_url = (
            "https://api.sandbox.paypal.com" if sandbox else "https://api.paypal.com"
        )

    def _do_charge(self, amount: float, description: str) -> dict:
        return {
            "status": "COMPLETED",
            "provider": "paypal",
            "id": "PAY_7tk12X",
            "amount": amount,
        }

    def refund(self, transaction_id: str) -> bool:
        self.log(f"PayPal refund for {transaction_id}")
        return True


class CryptoProcessor(PaymentProcessor):
    def __init__(self, api_key: str, wallet_address: str, network: str = "ETH"):
        super().__init__(api_key, currency=network)
        self.wallet_address = wallet_address
        self.gas_multiplier = 1.2

    def _do_charge(self, amount: float, description: str) -> dict:
        gas = amount * 0.003 * self.gas_multiplier
        return {
            "status": "mined",
            "provider": "crypto",
            "id": "0xabc123",
            "amount": amount,
            "gas": round(gas, 5),
        }

    def refund(self, transaction_id: str) -> bool:
        self.log("Crypto refunds require manual on-chain reversal")
        return False


def checkout(processor: PaymentProcessor, amount: float) -> dict:
    result = processor.charge(amount, "Order #8821")
    if result["status"] not in ("succeeded", "COMPLETED", "mined"):
        processor.refund(result["id"])
    return result
