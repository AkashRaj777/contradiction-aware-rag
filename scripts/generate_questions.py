"""
One-time script to build questions.jsonl from the user's labeled question lists.
Run: python scripts/generate_questions.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "eval" / "questions.jsonl"

NORMAL = [
    ("What is the minimum password length required under the new password policy?", ["password_policy_2025"]),
    ("What two-factor authentication methods are accepted for MFA?", ["mfa_requirements_2025"]),
    ("How many weeks of paid parental leave do primary caregivers receive?", ["parental_leave_2025"]),
    ("Is biometric authentication required for high-privilege accounts?", ["security_guidelines_2025"]),
    ("What is the annual PTO accrual rate for new hires under the 2025 PTO policy?", ["pto_policy_2025"]),
    ("How long is the probationary period before PTO begins accruing under the 2025 policy?", ["pto_policy_2025"]),
    ("Are contractors eligible for company-paid parental leave?", ["parental_leave_2025"]),
    ("What is the deadline to submit a refund request under the 2025 draft refund policy?", ["refund_policy_2025_draft"]),
    ("Are digital products eligible for refunds under the 2024 refund policy?", ["refund_policy_2024"]),
    ("Under the pre-COVID remote work policy, how many remote days per month were permitted?", ["remote_work_precovid"]),
    ("How often must employees rotate their passwords under the new security policy?", ["password_policy_2025"]),
    ("What is the secondary caregiver parental leave entitlement?", ["parental_leave_2025"]),
    ("Which department approves remote work exceptions under the 2025 policy?", ["remote_work_2025"]),
    ("What VPN client is required for remote access?", ["remote_work_2025", "security_guidelines_2025"]),
    ("Are personal devices allowed for accessing company systems under the new MFA requirements?", ["mfa_requirements_2025"]),
    ("What is the escalation path for reporting a security incident?", ["security_guidelines_2025"]),
    ("How are unused parental leave days handled at year-end?", ["parental_leave_2025"]),
    ("What documentation is required to process a refund?", ["refund_policy_2024"]),
    ("What is the stated reason for tightening security guidelines in the new policy?", ["security_guidelines_2025"]),
    ("Under the 2025 remote work policy, what is the core in-office attendance day?", ["remote_work_2025"]),
    ("What is the maximum file size allowed for password manager exports?", ["password_policy_2025"]),
    ("Are sabbaticals separate from regular PTO accruals?", ["pto_policy_2025"]),
    ("What is the notice period required for requesting extended PTO?", ["pto_policy_2025"]),
    ("Under the new password policy, can users reuse their last 5 passwords?", ["password_policy_2025"]),
    ("What is the refund method for original purchases made by credit card?", ["refund_policy_2024"]),
]

CONFLICT = [
    ("What is the refund window for standard purchases?", ["refund_policy_2024", "refund_policy_2025_draft"]),
    ("How many days per week must employees work from the office?", ["remote_work_precovid", "remote_work_2025"]),
    ("What is the minimum required password length?", ["password_policy_2024", "password_policy_2025"]),
    ("What is the annual PTO accrual rate for full-time employees?", ["pto_policy_2024", "pto_policy_2025"]),
    ("Are 30-day refunds currently allowed?", ["refund_policy_2024", "refund_policy_2025_draft"]),
    ("Can employees work fully remote on a permanent basis?", ["remote_work_precovid", "remote_work_2025"]),
    ("What is the mandatory password rotation frequency?", ["password_policy_2024", "password_policy_2025"]),
    ("When can new employees begin accruing PTO?", ["pto_policy_2024", "pto_policy_2025"]),
    ("Is fingerprint-based authentication acceptable as a primary MFA factor?", ["security_guidelines_2025", "mfa_requirements_2025"]),
    ("How many hybrid in-office days are permitted per week?", ["remote_work_precovid", "remote_work_2025"]),
    ("What is the refund processing turnaround time?", ["refund_policy_2024", "refund_policy_2025_draft"]),
    ("Are special characters required in passwords?", ["password_policy_2024", "password_policy_2025"]),
    ("What is the maximum PTO carryover into the following year?", ["pto_policy_2024", "pto_policy_2025"]),
    ("Under what conditions can a refund request be denied?", ["refund_policy_2024", "refund_policy_2025_draft"]),
    ("Is remote work the default arrangement for new hires?", ["remote_work_precovid", "remote_work_2025"]),
    ("How long must password history be retained to prevent reuse?", ["password_policy_2024", "password_policy_2025"]),
    ("What is the deadline to request a refund after the original purchase date?", ["refund_policy_2024", "refund_policy_2025_draft"]),
    ("Are managers permitted to override the in-office attendance requirement?", ["remote_work_2025"]),
    ("What is the official company-wide stance on full-time remote work?", ["remote_work_precovid", "remote_work_2025"]),
    ("Are partial refunds allowed for opened or partially used items?", ["refund_policy_2024", "refund_policy_2025_draft"]),
]

REFUSE = [
    "What is the company's stock price forecast for 2030?",
    "Who is the current CEO of the company?",
    "What is the dental insurance copay amount?",
    "How much is the annual end-of-year performance bonus?",
    "What was the company's revenue in Q4 2025?",
    "Are pets allowed in the office?",
    "What is the policy on accepting cryptocurrency for refunds?",
    "How many employees work in the Tokyo office?",
    "What is the maternity leave policy for employees based in Germany?",
    "What is the reimbursement rate for personal vehicle mileage?",
    "Does the company offer tuition reimbursement for graduate degrees?",
    "What is the overtime pay policy for salaried employees?",
    "Is there a signing or relocation bonus for new hires?",
    "What is the company's policy on jury duty leave?",
    "Are employees eligible for stock options after one year of service?",
]


def main():
    rows = []
    n = 1
    for q, docs in NORMAL:
        rows.append({
            "id": f"q{n:03d}",
            "question": q,
            "bucket": "normal",
            "expected_behavior": "answer",
            "gold_doc_ids": docs,
        })
        n += 1
    for q, docs in CONFLICT:
        rows.append({
            "id": f"q{n:03d}",
            "question": q,
            "bucket": "conflict",
            "expected_behavior": "present_both_sides",
            "gold_doc_ids": docs,
        })
        n += 1
    for q in REFUSE:
        rows.append({
            "id": f"q{n:03d}",
            "question": q,
            "bucket": "refuse",
            "expected_behavior": "refuse",
            "gold_doc_ids": [],
        })
        n += 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} questions to {OUT}")


if __name__ == "__main__":
    main()
