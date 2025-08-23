"""
Fraud Detection Simple Reflex Agent benchmark for AgenticEvals.

This benchmark tests a model's ability to classify multi-line, log-like
event sequences for a single user into exactly one of two categories:

- "fraud"
- "legitimate"

Scoring is strict binary (0/1):
- 1.0 if the model outputs exactly one of the allowed labels and it matches
  the ground truth label (case-insensitive exact token match, no extra text)
- 0.0 otherwise

Metrics recorded include execution_time (wall-clock), output_tokens, and
instruction-adherence diagnostics. All sensitive fields are pseudonymized
using generic identifiers (e.g., "IP Address A", "Device Type C").
"""

import time
import random
import re
from typing import List, Dict, Any

from ..benchmark.base import BaseBenchmark, Task, TaskResult, BenchmarkConfig, AgentType
from ..benchmark.registry import benchmark
from ..models.base import BaseModel, ModelResponse
from ..utils.logging import get_logger


logger = get_logger(__name__)


ALLOWED_LABELS = {"fraud", "legitimate"}
MAX_BENIGN_PREFIX_LINES = 8
MAX_BENIGN_SUFFIX_LINES = 7


@benchmark(
    name="fraud_detection",
    agent_type=AgentType.SIMPLE_REFLEX,
    description="Binary fraud detection on multi-line log-like scenarios"
)
class FraudDetectionBenchmark(BaseBenchmark):
    """
    Simple reflex agent benchmark using log-like fraud detection scenarios.

    The model must output exactly one of: "fraud" or "legitimate".
    Scoring is strict: any deviation from a single, exact label results in 0.
    """

    def get_tasks(self) -> List[Task]:
        """Construct 14 high-difficulty scenarios with subtle signals.

        Each scenario pertains to a single user (e.g., "User ID X").
        Optionally includes at most one signup event. All identifiers are generic.
        """
        tasks: List[Task] = []

        scenarios: List[Dict[str, Any]] = []

        def S(name: str, user_id: str, label: str, log: str) -> Dict[str, Any]:
            return {
                "name": name,
                "user_id": user_id,
                "label": label,
                "log": log.strip(),
            }

        # ------------------------------
        # Fraud scenarios (7)
        # ------------------------------

        scenarios.append(S(
            name="F1_CNP_HighValue_NewDevice_Proxy_Mismatch",
            user_id="User ID 7",
            label="fraud",
            log="""
event_id: E-1001 | event_time: 2025-06-01T05:42:15Z | event_type: login | user_id: User ID 7 | outcome: success | mfa_used: true | mfa_outcome: pass | robot_check_result: passed | ip_address: IP Address A | ip_country: Country A | region: Region Q | device_id: Device Type F | device_fingerprint: Fingerprint X-1 | known_device_flag: false | known_location_flag: partial | session_id: S-77 | prior_logins_from_ip_30d: 1
event_id: E-1002 | event_time: 2025-06-01T05:44:10Z | event_type: transaction | transaction_id: T-9001 | user_id: User ID 7 | account_id: Account ID 7A | amount: 480.00 | currency: Currency A | payment_method: card | channel: web | merchant_id: Merchant Q | card_network: Network A | card_bin: BIN X | card_last4: 0001 | issuer: Issuer X | auth_result: approved | auth_ip: IP Address A | ip_country: Country A | ip_asn: ASN Z | device_id: Device Type F | os: Operating System B | browser: Browser D | geohash: GH-AAA | location_confidence: medium | shipping_address: Address Type A | billing_address: Address Type A | time_of_day: early_morning
event_id: E-1003 | event_time: 2025-06-01T05:55:05Z | event_type: transaction | transaction_id: T-9002 | user_id: User ID 7 | account_id: Account ID 7A | amount: 510.00 | currency: Currency A | payment_method: card | channel: web | merchant_id: Merchant Q | card_network: Network A | card_bin: BIN X | card_last4: 0001 | issuer: Issuer X | auth_result: approved | auth_ip: IP Address A | ip_country: Country A | ip_asn: ASN Z | device_id: Device Type F | geohash: GH-AAB | location_confidence: medium | device_fingerprint_similarity: 0.92
            """
        ))

        scenarios.append(S(
            name="F2_CardTesting_SmallAuths_ThenBurst",
            user_id="User ID 5",
            label="fraud",
            log="""
event_id: E-2001 | event_time: 2025-07-11T08:00:01Z | event_type: transaction | transaction_id: T-5001 | user_id: User ID 5 | account_id: Account ID 5A | amount: 3.15 | currency: Currency A | payment_method: card | channel: api | merchant_id: Merchant R | card_network: Network A | card_bin: BIN Y | card_last4: 2222 | issuer: Issuer X | auth_result: approved | auth_ip: IP Address B | ip_country: Country A | ip_asn: ASN Y | device_id: Device Type C
event_id: E-2002 | event_time: 2025-07-11T08:03:40Z | event_type: transaction | transaction_id: T-5002 | user_id: User ID 5 | account_id: Account ID 5A | amount: 4.20 | currency: Currency A | payment_method: card | channel: api | merchant_id: Merchant R | card_network: Network A | card_bin: BIN Y | card_last4: 2222 | issuer: Issuer X | auth_result: approved | auth_ip: IP Address B | ip_country: Country A | ip_asn: ASN Y | device_id: Device Type C
            """
        ))

        scenarios.append(S(
            name="F3_ATO_NewDevice_RegionShift_NoMFA",
            user_id="User ID 3",
            label="fraud",
            log="""
event_id: E-3001 | event_time: 2025-05-21T04:03:10Z | event_type: login | user_id: User ID 3 | outcome: success | mfa_used: true | mfa_outcome: pass | robot_check_result: passed | ip_address: IP Address C | ip_country: Country B | region: Region Q | device_id: Device Type G | device_fingerprint: Fingerprint Y-1 | known_device_flag: false | known_location_flag: partial | session_id: S-302
event_id: E-3002 | event_time: 2025-05-21T04:11:33Z | event_type: transaction | transaction_id: T-3007 | user_id: User ID 3 | account_id: Account ID 3A | amount: 210.00 | currency: Currency A | payment_method: card | channel: web | merchant_id: Merchant T | card_network: Network A | card_bin: BIN Z | card_last4: 1010 | issuer: Issuer X | auth_result: approved | auth_ip: IP Address C | ip_country: Country B | device_id: Device Type G | location_confidence: medium | shipping_address: Address Type A | billing_address: Address Type A | time_of_day: early_morning
            """
        ))

        scenarios.append(S(
            name="F4_API_Wire_NewBeneficiary_SuspiciousASN",
            user_id="User ID 9",
            label="fraud",
            log="""
event_id: E-4001 | event_time: 2025-04-03T09:10:00Z | event_type: login | user_id: User ID 9 | outcome: success | mfa_used: true | mfa_outcome: pass | robot_check_result: passed | ip_address: IP Address D | ip_country: Country A | device_id: Device Type H | device_fingerprint: Fingerprint Z | known_device_flag: true | known_location_flag: true | session_id: S-900
event_id: E-4002 | event_time: 2025-04-03T09:18:12Z | event_type: transaction | transaction_id: T-9100 | user_id: User ID 9 | account_id: Account ID 9A | amount: 1800.00 | currency: Currency A | payment_method: bank | channel: api | merchant_id: Merchant U | auth_result: pending | auth_ip: IP Address D | ip_country: Country A | device_id: Device Type H | beneficiary_age_days: 0 | location_confidence: medium
            """
        ))

        scenarios.append(S(
            name="F5_OTP_Reset_EmailChange_SameIP_NewDevice",
            user_id="User ID 2",
            label="fraud",
            log="""
event_id: E-5001 | event_time: 2025-08-12T05:12:03Z | event_type: account_change | change_type: password_reset | user_id: User ID 2 | outcome: success | ip_address: IP Address E | device_id: Device Type C | session_id: S-22 | recovery_channel: email
event_id: E-5002 | event_time: 2025-08-12T05:14:12Z | event_type: login | user_id: User ID 2 | outcome: success | mfa_used: true | mfa_outcome: pass | robot_check_result: passed | ip_address: IP Address E | device_id: Device Type C | device_fingerprint: Fingerprint A-1 | known_device_flag: false | session_id: S-23 | time_of_day: early_morning
            """
        ))

        scenarios.append(S(
            name="F6_Synthetic_Signup_ImmediateWalletTopups",
            user_id="User ID 11",
            label="fraud",
            log="""
event_id: E-6000 | event_time: 2025-03-10T09:00:00Z | event_type: signup | signup_id: SG-11 | user_id: User ID 11 | ip_address: IP Address F | ip_country: Country A | device_id: Device Type D | device_fingerprint: Fingerprint C | robot_check_result: passed | email_domain: Domain B (new-ish)
event_id: E-6001 | event_time: 2025-03-10T09:05:12Z | event_type: transaction | transaction_id: T-6100 | user_id: User ID 11 | account_id: Account ID 11A | amount: 180.00 | currency: Currency A | payment_method: wallet | channel: api | merchant_id: Merchant V | auth_result: approved | ip_address: IP Address F | device_id: Device Type D | location_confidence: medium
            """
        ))

        scenarios.append(S(
            name="F7_MerchantCluster_NearDuplicates_IPRotation",
            user_id="User ID 10",
            label="fraud",
            log="""
event_id: E-7001 | event_time: 2025-02-02T16:00:00Z | event_type: transaction | transaction_id: T-7100 | user_id: User ID 10 | account_id: Account ID 10A | amount: 87.90 | currency: Currency A | payment_method: card | channel: web | merchant_id: Merchant Q-1 | card_network: Network A | card_bin: BIN W | card_last4: 3333 | issuer: Issuer X | auth_result: declined | auth_ip: IP Address G | ip_country: Country A | ip_asn: ASN Y | device_id: Device Type B | velocity_1h: 1
event_id: E-7002 | event_time: 2025-02-02T16:03:12Z | event_type: transaction | transaction_id: T-7101 | user_id: User ID 10 | account_id: Account ID 10A | amount: 88.05 | currency: Currency A | payment_method: card | channel: web | merchant_id: Merchant Q-2 | card_network: Network A | card_bin: BIN W | card_last4: 3333 | issuer: Issuer X | auth_result: approved | auth_ip: IP Address H | ip_country: Country A | ip_asn: ASN Y | device_id: Device Type B | velocity_1h: 2
event_id: E-7003 | event_time: 2025-02-02T16:09:25Z | event_type: transaction | transaction_id: T-7102 | user_id: User ID 10 | account_id: Account ID 10A | amount: 88.50 | currency: Currency A | payment_method: card | channel: web | merchant_id: Merchant Q-3 | card_network: Network A | card_bin: BIN W | card_last4: 3333 | issuer: Issuer X | auth_result: declined | auth_ip: IP Address G | ip_country: Country A | ip_asn: ASN Y | device_id: Device Type B
            """
        ))

        scenarios.append(S(
            name="F8_Impossible_Country_Change_NoVPN",
            user_id="User ID 15",
            label="fraud",
            log="""
event_id: E-9101 | event_time: 2025-06-10T12:00:00Z | event_type: login | user_id: User ID 15 | outcome: success | mfa_used: true | mfa_outcome: pass | robot_check_result: passed | ip_address: IP Address R | ip_country: Country A | region: Region A | device_id: Device Type A | device_fingerprint: Fingerprint J | known_device_flag: true | known_location_flag: true | session_id: S-150
event_id: E-9102 | event_time: 2025-06-10T12:07:30Z | event_type: transaction | transaction_id: T-9102 | user_id: User ID 15 | account_id: Account ID 15A | amount: 260.00 | currency: Currency A | payment_method: card | channel: mobile | merchant_id: Merchant U | card_network: Network A | card_bin: BIN U | card_last4: 1212 | issuer: Issuer X | auth_result: approved | auth_ip: IP Address S | ip_country: Country C | ip_asn: ASN X | proxy_vpn_flag: false | device_id: Device Type A | device_fingerprint_similarity: 0.98 | geo_lat: 55.000 | geo_lon: 12.000 | location_confidence: high | shipping_address: Address Type A | billing_address: Address Type A
            """
        ))

        # Two transactions at the exact same timestamp (subtle impossibility)
        scenarios.append(S(
            name="F9_Simultaneous_TwoTransactions_SameTimestamp",
            user_id="User ID 16",
            label="fraud",
            log="""
event_id: E-9201 | event_time: 2025-07-01T10:15:30Z | event_type: transaction | transaction_id: T-9201 | user_id: User ID 16 | account_id: Account ID 16A | amount: 48.90 | currency: Currency A | payment_method: card | channel: web | merchant_id: Merchant P | card_network: Network A | card_bin: BIN V | card_last4: 2323 | issuer: Issuer X | auth_result: approved | auth_ip: IP Address T | ip_country: Country A | ip_asn: ASN X | device_id: Device Type A | location_confidence: high | billing_address: Address Type A | shipping_address: Address Type A
event_id: E-9202 | event_time: 2025-07-01T10:15:30Z | event_type: transaction | transaction_id: T-9202 | user_id: User ID 16 | account_id: Account ID 16A | amount: 52.10 | currency: Currency A | payment_method: card | channel: web | merchant_id: Merchant Q | card_network: Network A | card_bin: BIN V | card_last4: 2323 | issuer: Issuer X | auth_result: approved | auth_ip: IP Address T | ip_country: Country A | ip_asn: ASN X | device_id: Device Type A | location_confidence: high | billing_address: Address Type A | shipping_address: Address Type A
            """
        ))

        # One login and one transaction at the almost exact same timestamp (subtle impossibility)
        scenarios.append(S(
            name="F10_Simultaneous_Login_And_Transaction",
            user_id="User ID 17",
            label="fraud",
            log="""
event_id: E-9301 | event_time: 2025-08-05T18:22:05Z | event_type: login | user_id: User ID 17 | outcome: success | mfa_used: true | mfa_outcome: pass | robot_check_result: passed | ip_address: IP Address U | ip_country: Country A | region: Region C | device_id: Device Type A | device_fingerprint: Fingerprint K | known_device_flag: true | known_location_flag: true | session_id: S-170
event_id: E-9302 | event_time: 2025-08-05T18:22:06Z | event_type: transaction | transaction_id: T-9302 | user_id: User ID 17 | account_id: Account ID 17A | amount: 95.00 | currency: Currency A | payment_method: card | channel: web | merchant_id: Merchant R | card_network: Network A | card_bin: BIN T | card_last4: 3434 | issuer: Issuer X | auth_result: approved | auth_ip: IP Address U | ip_country: Country A | ip_asn: ASN X | device_id: Device Type A | location_confidence: high | billing_address: Address Type A | shipping_address: Address Type A
            """
        ))

        # Multiple countries within the same hour (single subtle clue: rapid geo change)
        scenarios.append(S(
            name="F11_MultiCountry_SameHour_NoVPN",
            user_id="User ID 18",
            label="fraud",
            log="""
event_id: E-9401 | event_time: 2025-09-12T13:02:10Z | event_type: login | user_id: User ID 18 | outcome: success | mfa_used: true | mfa_outcome: pass | robot_check_result: passed | ip_address: IP Address V | ip_country: Country A | region: Region A | device_id: Device Type A | device_fingerprint: Fingerprint L | known_device_flag: true | session_id: S-180
event_id: E-9402 | event_time: 2025-09-12T13:10:30Z | event_type: transaction | transaction_id: T-9402 | user_id: User ID 18 | account_id: Account ID 18A | amount: 64.20 | currency: Currency A | payment_method: card | channel: web | merchant_id: Merchant P | card_network: Network A | card_bin: BIN S | card_last4: 4545 | issuer: Issuer X | auth_result: approved | auth_ip: IP Address W | ip_country: Country B | ip_asn: ASN X | device_id: Device Type A | device_fingerprint_similarity: 0.99 | location_confidence: high
event_id: E-9403 | event_time: 2025-09-12T13:31:05Z | event_type: login | user_id: User ID 18 | outcome: success | mfa_used: true | mfa_outcome: pass | robot_check_result: passed | ip_address: IP Address X | ip_country: Country C | region: Region C | device_id: Device Type A | device_fingerprint: Fingerprint L-1 | known_device_flag: true | session_id: S-181
event_id: E-9404 | event_time: 2025-09-12T13:48:50Z | event_type: transaction | transaction_id: T-9404 | user_id: User ID 18 | account_id: Account ID 18A | amount: 72.80 | currency: Currency A | payment_method: card | channel: mobile | merchant_id: Merchant Q | card_network: Network A | card_bin: BIN S | card_last4: 4545 | issuer: Issuer X | auth_result: approved | auth_ip: IP Address Y | ip_country: Country D | ip_asn: ASN X | device_id: Device Type A | device_fingerprint_similarity: 0.98 | location_confidence: high
            """
        ))

        # ------------------------------
        # Legitimate scenarios (7)
        # ------------------------------

        scenarios.append(S(
            name="L1_Recurring_Subscription_KnownDevice_StableGeo",
            user_id="User ID 1",
            label="legitimate",
            log="""
event_id: E-8001 | event_time: 2025-01-01T09:00:00Z | event_type: login | user_id: User ID 1 | outcome: success | mfa_used: true | mfa_outcome: pass | robot_check_result: passed | ip_address: IP Address J | ip_country: Country A | device_id: Device Type A | device_fingerprint: Fingerprint A | known_device_flag: true | known_location_flag: true | session_id: S-1
event_id: E-8002 | event_time: 2025-01-01T09:02:00Z | event_type: transaction | transaction_id: T-8000 | user_id: User ID 1 | account_id: Account ID 1A | amount: 9.99 | currency: Currency A | payment_method: wallet | channel: web | merchant_id: Merchant S | auth_result: approved | auth_ip: IP Address J | ip_country: Country A | ip_asn: ASN Y | proxy_vpn_flag: false | device_id: Device Type A | geo_lat: 20.001 | geo_lon: 20.002 | geohash: GH-CCC | location_confidence: high | velocity_24h: 1
            """
        ))

        scenarios.append(S(
            name="L2_Grocery_Local_KnownDevice_NormalAggregates",
            user_id="User ID 4",
            label="legitimate",
            log="""
event_id: E-8101 | event_time: 2025-01-07T17:31:00Z | event_type: login | user_id: User ID 4 | outcome: success | mfa_used: true | mfa_outcome: pass | robot_check_result: passed | ip_address: IP Address K | ip_country: Country A | device_id: Device Type A | device_fingerprint: Fingerprint D | known_device_flag: true | known_location_flag: true | session_id: S-10
event_id: E-8102 | event_time: 2025-01-07T17:35:25Z | event_type: transaction | transaction_id: T-8100 | user_id: User ID 4 | account_id: Account ID 4A | amount: 76.10 | currency: Currency A | payment_method: card | channel: mobile | merchant_id: Merchant P | card_network: Network A | card_bin: BIN T | card_last4: 4444 | issuer: Issuer X | auth_result: approved | auth_ip: IP Address K | ip_country: Country A | ip_asn: ASN Y | proxy_vpn_flag: false | device_id: Device Type A | geo_lat: 30.111 | geo_lon: 30.112 | geohash: GH-DDD | location_confidence: high | velocity_24h: 1
            """
        ))

        scenarios.append(S(
            name="L3_PayrollAndInternalTransfer_NoAnomalies",
            user_id="User ID 8",
            label="legitimate",
            log="""
event_id: E-8201 | event_time: 2025-02-01T12:00:00Z | event_type: transaction | transaction_id: T-8200 | user_id: User ID 8 | account_id: Account ID 8A | amount: 2400.00 | currency: Currency A | payment_method: bank | channel: api | merchant_id: Merchant N | auth_result: approved | auth_ip: IP Address L | ip_country: Country A | ip_asn: ASN X | proxy_vpn_flag: false | device_id: Device Type A | descriptor: payroll | velocity_24h: 1
event_id: E-8202 | event_time: 2025-02-01T12:05:00Z | event_type: transaction | transaction_id: T-8201 | user_id: User ID 8 | account_id: Account ID 8B | amount: 800.00 | currency: Currency A | payment_method: bank | channel: api | merchant_id: Merchant N | auth_result: approved | auth_ip: IP Address L | ip_country: Country A | ip_asn: ASN X | proxy_vpn_flag: false | device_id: Device Type A | descriptor: internal_transfer | velocity_24h: 2
            """
        ))

        scenarios.append(S(
            name="L4_TravelDay_ItineraryAligned_GeoShiftKnown",
            user_id="User ID 6",
            label="legitimate",
            log="""
event_id: E-8301 | event_time: 2025-03-15T05:30:00Z | event_type: login | user_id: User ID 6 | outcome: success | mfa_used: true | mfa_outcome: pass | robot_check_result: passed | ip_address: IP Address M | ip_country: Country A | device_id: Device Type A | device_fingerprint: Fingerprint E | known_device_flag: true | known_location_flag: true | session_id: S-60
event_id: E-8302 | event_time: 2025-03-15T06:00:00Z | event_type: transaction | transaction_id: T-8300 | user_id: User ID 6 | account_id: Account ID 6A | amount: 320.00 | currency: Currency A | payment_method: card | channel: mobile | merchant_id: Merchant Air | card_network: Network A | card_bin: BIN U | card_last4: 5555 | issuer: Issuer X | auth_result: approved | auth_ip: IP Address M | ip_country: Country A | ip_asn: ASN Y | proxy_vpn_flag: false | device_id: Device Type A | descriptor: airline_booking (prior) | location_confidence: high
event_id: E-8303 | event_time: 2025-03-15T15:00:00Z | event_type: transaction | transaction_id: T-8301 | user_id: User ID 6 | account_id: Account ID 6A | amount: 140.00 | currency: Currency A | payment_method: card | channel: mobile | merchant_id: Merchant Hotel | card_network: Network A | card_bin: BIN U | card_last4: 5555 | issuer: Issuer X | auth_result: approved | auth_ip: IP Address N | ip_country: Country B | ip_asn: ASN Y | proxy_vpn_flag: false | device_id: Device Type A | descriptor: hotel_checkin | location_confidence: high | known_location_flag: true
            """
        ))

        scenarios.append(S(
            name="L5_NewMerchant_MatchAddresses_KnownDevice",
            user_id="User ID 12",
            label="legitimate",
            log="""
event_id: E-8401 | event_time: 2025-04-21T11:10:00Z | event_type: login | user_id: User ID 12 | outcome: success | mfa_used: true | mfa_outcome: pass | robot_check_result: passed | ip_address: IP Address O | ip_country: Country A | device_id: Device Type A | device_fingerprint: Fingerprint F | known_device_flag: true | known_location_flag: true | session_id: S-120
event_id: E-8402 | event_time: 2025-04-21T11:12:30Z | event_type: transaction | transaction_id: T-8400 | user_id: User ID 12 | account_id: Account ID 12A | amount: 2400.00 | currency: Currency A | payment_method: card | channel: web | merchant_id: Merchant New | card_network: Network A | card_bin: BIN S | card_last4: 6666 | issuer: Issuer X | auth_result: approved | auth_ip: IP Address O | ip_country: Country A | ip_asn: ASN X | device_id: Device Type A | shipping_address: Address Type A | billing_address: Address Type A | location_confidence: high | prior_large_orders_90d: 2 | three_ds_result: successful
            """
        ))

        scenarios.append(S(
            name="L6_OneTimeDonation_SmallAmount_ConsistentSignals",
            user_id="User ID 13",
            label="legitimate",
            log="""
event_id: E-8501 | event_time: 2025-06-05T19:40:00Z | event_type: login | user_id: User ID 13 | outcome: success | mfa_used: true | mfa_outcome: pass | robot_check_result: passed | ip_address: IP Address P | ip_country: Country A | device_id: Device Type A | device_fingerprint: Fingerprint G | known_device_flag: true | known_location_flag: true | session_id: S-130
event_id: E-8502 | event_time: 2025-06-05T19:41:45Z | event_type: transaction | transaction_id: T-8500 | user_id: User ID 13 | account_id: Account ID 13A | amount: 15.00 | currency: Currency A | payment_method: card | channel: web | merchant_id: Merchant Donate | card_network: Network A | card_bin: BIN R | card_last4: 7777 | issuer: Issuer X | auth_result: approved | auth_ip: IP Address P | ip_country: Country A | ip_asn: ASN X | proxy_vpn_flag: false | device_id: Device Type A | location_confidence: high | velocity_24h: 1
            """
        ))

        scenarios.append(S(
            name="L7_Signup_ThenSmallPurchase_SameDevice_StableIP",
            user_id="User ID 14",
            label="legitimate",
            log="""
event_id: E-8600 | event_time: 2025-07-20T10:00:00Z | event_type: signup | signup_id: SG-14 | user_id: User ID 14 | ip_address: IP Address Q | ip_country: Country A | device_id: Device Type A | device_fingerprint: Fingerprint H | robot_check_result: passed | email_domain: Domain A (known)
event_id: E-8601 | event_time: 2025-07-20T10:10:00Z | event_type: login | user_id: User ID 14 | outcome: success | mfa_used: true | mfa_outcome: pass | robot_check_result: passed | ip_address: IP Address Q | device_id: Device Type A | known_device_flag: true | known_location_flag: true | session_id: S-140
event_id: E-8602 | event_time: 2025-07-20T10:12:30Z | event_type: transaction | transaction_id: T-8600 | user_id: User ID 14 | account_id: Account ID 14A | amount: 12.00 | currency: Currency A | payment_method: card | channel: web | merchant_id: Merchant Small | card_network: Network A | card_bin: BIN Q | card_last4: 8888 | issuer: Issuer X | auth_result: approved | auth_ip: IP Address Q | ip_country: Country A | ip_asn: ASN X | proxy_vpn_flag: false | device_id: Device Type A | location_confidence: high | velocity_24h: 1
            """
        ))

        # ------------------------------
        # Build Task objects (total 14 scenarios)
        # ------------------------------

        for i, sc in enumerate(scenarios, start=1):
            # Deterministic randomness per scenario using seed 123
            prefix_rng = random.Random(123 + i * 2)
            suffix_rng = random.Random(123 + i * 2 + 1)
            interleave_rng = random.Random(123 + i * 3)

            # Vary the benign prefix/suffix counts deterministically per scenario within caps
            low_prefix = max(6, MAX_BENIGN_PREFIX_LINES // 2)
            low_suffix = max(5, MAX_BENIGN_SUFFIX_LINES // 2)
            prefix_count = prefix_rng.randint(low_prefix, MAX_BENIGN_PREFIX_LINES)
            suffix_count = suffix_rng.randint(low_suffix, MAX_BENIGN_SUFFIX_LINES)

            # Generate benign noise blocks with deterministic randomized amounts and unique sequencing
            seq_counter = (i - 1) * 1000
            benign_prefix, seq_counter = self._generate_benign_log_lines(sc["user_id"], prefix_count, prefix_rng, seq_counter)
            # Interleave benign lines between core suspicious/legitimate events to increase difficulty
            interleaved_core, seq_counter = self._interleave_benign_lines(sc["log"], sc["user_id"], interleave_rng, seq_counter)
            benign_suffix, seq_counter = self._generate_benign_log_lines(sc["user_id"], suffix_count, suffix_rng, seq_counter)
            combined_log = (benign_prefix + "\n" + interleaved_core + "\n" + benign_suffix).strip()

            task = Task(
                task_id=f"fraud_simple_{i}",
                name=f"Fraud Detection: {sc['name']}",
                description="Binary fraud classification on multi-line logs (strict format)",
                prompt=self._create_prompt(combined_log, sc["user_id"]),
                expected_output=sc["label"],
                evaluation_criteria={
                    "exact_match": True,
                    "case_insensitive": True,
                    "allowed_labels": list(ALLOWED_LABELS),
                },
                metadata={
                    "user_id": sc["user_id"],
                    "expected_response": sc["label"],
                    "difficulty": "hard",
                    "lines_in_log": len(combined_log.splitlines()),
                }
            )
            tasks.append(task)

        # Shuffle task order to avoid contiguous blocks of the same label
        shuffle_seed = 123
        random.Random(shuffle_seed).shuffle(tasks)

        return tasks

    def _create_prompt(self, log_content: str, user_id: str) -> str:
        """Create the instruction + log prompt."""
        return f"""Fraud Detection Classification

Privacy Notice:
All sensitive fields are pseudonymized (e.g., "IP Address A", "Device Type C", "User ID 1", "Country A").

Categories (choose exactly one):
- "fraud"
- "legitimate"

Instructions:
- Read the multi-line log carefully. It pertains to exactly one user (e.g., "{user_id}").
- If signals indicate fraud or account takeover risk, output "fraud".
- If signals indicate normal behavior, output "legitimate".
- Output exactly one category label. No explanations or extra text.

LOG (User: {user_id}):
{log_content}

Your response:"""

    async def evaluate_task(self, task: Task, model: BaseModel) -> TaskResult:
        """Evaluate a single fraud detection task with strict formatting rules."""
        start_time = time.time()

        try:
            model_response = await model.generate(task.prompt)
            execution_time = time.time() - start_time

            score = self.calculate_score(task, model_response)
            success = score == 1.0

            metrics = self._calculate_detailed_metrics(task, model_response)
            metrics["execution_time"] = execution_time
            metrics["score"] = score

            return TaskResult(
                task_id=task.task_id,
                task_name=task.name,
                agent_type=self.agent_type,
                success=success,
                score=score,
                metrics=metrics,
                model_response=model_response,
                execution_time=execution_time,
                metadata=task.metadata
            )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Error evaluating task {task.task_id}: {e}")

            return TaskResult(
                task_id=task.task_id,
                task_name=task.name,
                agent_type=self.agent_type,
                success=False,
                score=0.0,
                metrics={"execution_time": execution_time},
                execution_time=execution_time,
                error_message=str(e),
                metadata=task.metadata
            )

    def calculate_score(self, task: Task, model_response: ModelResponse) -> float:
        """Strict 0/1 scoring based on exact single-label match.

        Rules:
        - Response must be exactly one token equal to one of the allowed labels
          (case-insensitive). No punctuation, no extra words.
        - If it matches the expected label => 1.0, else 0.0.
        """
        if not model_response or not model_response.text:
            return 0.0

        text = model_response.text.strip()

        # Enforce exactly one token (no spaces)
        if not text or (" " in text.strip()):
            return 0.0

        normalized = text.lower().strip()

        # Enforce no trailing punctuation or extra characters beyond the token
        if not re.fullmatch(r"[a-z_]+", normalized):
            return 0.0

        if normalized not in ALLOWED_LABELS:
            return 0.0

        expected = (task.expected_output or "").strip().lower()
        return 1.0 if normalized == expected else 0.0

    def _calculate_detailed_metrics(self, task: Task, model_response: ModelResponse) -> Dict[str, Any]:
        """Collect diagnostics for analysis (not used in scoring)."""
        response_text = model_response.text if (model_response and model_response.text) else ""
        stripped = response_text.strip()
        normalized = stripped.lower()

        word_count = len(stripped.split()) if stripped else 0
        char_count = len(stripped)

        is_single_token = (word_count == 1)
        token_is_valid = normalized in ALLOWED_LABELS
        follows_instructions = is_single_token and token_is_valid

        expected = (task.expected_output or "").strip().lower()
        exact_match = follows_instructions and (normalized == expected)

        cleaned_response = re.sub(r"[^a-z_]", "", normalized)

        return {
            "word_count": word_count,
            "char_count": char_count,
            "follows_instructions": follows_instructions,
            "is_valid_response": token_is_valid,
            "exact_match": exact_match,
            "cleaned_response": cleaned_response,
            "response_latency": model_response.latency if model_response else None,
            "output_tokens": model_response.completion_tokens if model_response else 0,
        }

    def _generate_benign_log_lines(self, user_id: str, count: int, rng: random.Random, seq_start: int):
        """Generate benign (non-suspicious) lines with unique ids and spaced timestamps.

        Returns (log_block_str, next_seq_counter).
        """
        lines = []
        for i in range(count):
            seq = seq_start + i
            minutes_since_start = seq * 5
            hour = (minutes_since_start // 60) % 24
            minute = minutes_since_start % 60
            second = rng.randint(0, 59)
            if i % 2 == 0:
                lines.append(
                    (
                        f"event_id: E-B{1000 + seq} | event_time: 2025-01-01T{hour:02d}:{minute:02d}:{second:02d}Z | "
                        f"event_type: login | user_id: {user_id} | outcome: success | mfa_used: true | "
                        f"mfa_outcome: pass | robot_check_result: passed "
                        f"device_id: Device Type A | "
                        f"known_device_flag: true | "
                        f"known_location_flag: true"
                    )
                )
            else:
                amount = round(rng.uniform(5.00, 25.00), 2)
                lines.append(
                    (
                        f"event_id: E-B{1000 + seq} | event_time: 2025-01-01T{hour:02d}:{minute:02d}:{second:02d}Z | "
                        f"event_type: transaction | transaction_id: T-B{2000 + seq} | user_id: {user_id} | "
                        f"account_id: Account ID benign | amount: {amount:.2f} | currency: Currency A | "
                        f"payment_method: card | channel: web | merchant_id: Merchant E | "
                        f"card_network: Network A | card_bin: BIN benign | card_last4: 9999 | issuer: Issuer X | "
                        f"auth_result: approved | proxy_vpn_flag: false | device_id: Device Type A | "
                        f"geohash: GH-BENIGN | location_confidence: high"
                    )
                )
        return "\n".join(lines), (seq_start + count)

    def _interleave_benign_lines(self, core_log: str, user_id: str, rng: random.Random, seq_start: int):
        """Interleave 1-2 benign lines between each pair of core log lines deterministically.

        Returns (log_block_str, next_seq_counter). Keeps token growth controlled by capping the number
        of inserted lines per gap.
        """
        core_lines = [ln for ln in core_log.strip().splitlines() if ln.strip()]
        if not core_lines:
            return core_log.strip(), seq_start

        output_lines = []
        seq = seq_start
        for idx, ln in enumerate(core_lines):
            output_lines.append(ln)
            if idx < len(core_lines) - 1:
                insert_n = rng.randint(1, 2)
                benign_block, seq = self._generate_benign_log_lines(user_id, insert_n, rng, seq)
                for b in benign_block.splitlines():
                    if b.strip():
                        output_lines.append(b)
        return "\n".join(output_lines), seq


