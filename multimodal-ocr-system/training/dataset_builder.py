"""
Phase 2 — Dataset Builder
Generates a rich synthetic labelled dataset for fine-tuning BERT.

Labels:
  0 = medical_prescription
  1 = lab_report
  2 = legal_contract
  3 = affidavit
  4 = invoice
  5 = unknown

Each sample is a realistic document text excerpt.
"""

import json
import random
import os
from dataclasses import dataclass, field, asdict
from typing import List

random.seed(42)

# ─── Label map ────────────────────────────────────────────────────────────────
LABEL2ID = {
    "medical_prescription": 0,
    "lab_report": 1,
    "legal_contract": 2,
    "affidavit": 3,
    "invoice": 4,
    "unknown": 5,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# ─── Template pools ───────────────────────────────────────────────────────────

PATIENT_NAMES   = ["John Doe", "Jane Smith", "Robert Kumar", "Priya Patel",
                   "Michael Chen", "Sarah Williams", "David Lee", "Anita Sharma"]
DOCTOR_NAMES    = ["Dr. Sarah Williams", "Dr. A. Kumar", "Dr. P. Mehta",
                   "Dr. R. Singh", "Dr. Emily Stone", "Dr. James Carter"]
DRUGS           = ["Amoxicillin", "Ibuprofen", "Metformin", "Lisinopril",
                   "Atorvastatin", "Paracetamol", "Aspirin", "Omeprazole",
                   "Amlodipine", "Metoprolol", "Prednisone", "Warfarin"]
DOSAGES         = ["500mg", "250mg", "10mg", "20mg", "40mg", "100mg", "400mg"]
FREQUENCIES     = ["once daily", "twice daily", "thrice daily",
                   "every 8 hours", "every 12 hours", "as needed"]
COMPANIES       = ["XYZ Technology Solutions Pvt. Ltd.",
                   "ABC Retail Corporation", "GlobalTech Inc.",
                   "Sunrise Pharmaceuticals", "LexCorp Legal Services",
                   "Omega Finance Group"]
JURISDICTIONS   = ["State of California", "State of New York",
                   "State of Maharashtra", "Province of Ontario",
                   "District of Columbia"]
LAB_TESTS       = ["Hemoglobin", "WBC", "RBC", "Platelet", "HbA1c",
                   "Fasting Glucose", "Creatinine", "Bilirubin", "Cholesterol"]
LAB_UNITS       = ["g/dL", "K/uL", "M/uL", "%", "mg/dL", "mmol/L"]
INVOICE_ITEMS   = ["Software License", "Consulting Services", "Legal Advisory",
                   "Medical Equipment", "Office Supplies", "IT Support"]
DATES           = ["01/03/2024", "15/06/2024", "22/09/2024", "10/12/2024",
                   "05/01/2025", "28/02/2025"]
AMOUNTS         = ["USD 5,000", "USD 12,500", "INR 75,000", "EUR 8,200",
                   "USD 1,500", "GBP 3,000"]


# ─── Per-label text generators ────────────────────────────────────────────────

def _gen_prescription() -> str:
    patient = random.choice(PATIENT_NAMES)
    doctor  = random.choice(DOCTOR_NAMES)
    date    = random.choice(DATES)
    drug1   = random.choice(DRUGS)
    drug2   = random.choice([d for d in DRUGS if d != drug1])
    dose1   = random.choice(DOSAGES)
    dose2   = random.choice(DOSAGES)
    freq1   = random.choice(FREQUENCIES)
    freq2   = random.choice(FREQUENCIES)
    days    = random.choice([5, 7, 10, 14])
    refills = random.choice([0, 1, 2])

    templates = [
        f"""MEDICAL PRESCRIPTION
Patient Name: {patient}
Date: {date}
Prescribed by: {doctor}

Rx:
1. {drug1} {dose1} — {freq1} for {days} days
2. {drug2} {dose2} — {freq2} for {days} days

Diagnosis: Acute upper respiratory infection
Refills: {refills}
Dispensed by licensed pharmacy only.""",

        f"""City General Hospital — Outpatient Prescription
Patient: {patient}   Consulting Physician: {doctor}
Date of Issue: {date}

Medications Prescribed:
- {drug1} {dose1}: Take {freq1} with food
- {drug2} {dose2}: Take {freq2} at bedtime

Special Instructions: Avoid alcohol. Complete full course.
Valid for 30 days. Refills allowed: {refills}""",

        f"""Rx
Pt: {patient}
Rx Date: {date}
Dr: {doctor}

{drug1} {dose1} — Sig: {freq1} x {days}d — Qty: {days * 2}
{drug2} {dose2} — Sig: {freq2} x {days}d — Qty: {days}

No substitution. Patient counselled. Refills: {refills}.""",
    ]
    return random.choice(templates)


def _gen_lab_report() -> str:
    patient  = random.choice(PATIENT_NAMES)
    doctor   = random.choice(DOCTOR_NAMES)
    date     = random.choice(DATES)
    lab_id   = f"LAB-{random.randint(1000, 9999)}"

    # pick 3-5 random tests
    n_tests  = random.randint(3, 5)
    tests    = random.sample(LAB_TESTS, n_tests)
    rows = []
    for test in tests:
        val  = round(random.uniform(3.0, 15.0), 1)
        unit = random.choice(LAB_UNITS)
        flag = random.choice(["NORMAL", "LOW", "HIGH", "CRITICAL"])
        rows.append(f"  {test:<25} {val} {unit:<10} {flag}")

    results_block = "\n".join(rows)

    templates = [
        f"""PATHCARE DIAGNOSTICS — LABORATORY REPORT
Patient: {patient}    Sample ID: {lab_id}
Referred By: {doctor}    Date: {date}

TEST RESULTS:
{results_block}

Laboratory Director: Dr. Certified Pathologist
Report verified and digitally signed.""",

        f"""Clinical Laboratory Services
Report Date: {date}    Accession: {lab_id}
Patient Name: {patient}    Physician: {doctor}

Complete Metabolic Panel & CBC:
{results_block}

Note: Values outside reference range are flagged.
Interpretation should be done in clinical context.""",

        f"""LAB REPORT — {lab_id}
Patient: {patient}
Ordering Provider: {doctor}    Collection Date: {date}

{results_block}

Specimen: Venous blood collected under sterile conditions.
Findings discussed with ordering physician.""",
    ]
    return random.choice(templates)


def _gen_legal_contract() -> str:
    party_a = random.choice(COMPANIES)
    party_b = random.choice([c for c in COMPANIES if c != party_a])
    date    = random.choice(DATES)
    amount  = random.choice(AMOUNTS)
    juris   = random.choice(JURISDICTIONS)
    notice  = random.choice([15, 30, 60])

    templates = [
        f"""SERVICE AGREEMENT
This Agreement is entered into on {date} between:
{party_a} (hereinafter "Service Provider")
AND
{party_b} (hereinafter "Client")

WHEREAS the parties desire to set forth the terms of engagement,

1. SERVICES: Service Provider shall deliver software consulting services.
2. PAYMENT: Client shall pay {amount} per month within 30 days of invoice.
3. CONFIDENTIALITY: Both parties shall maintain strict confidentiality.
4. TERMINATION: Either party may terminate with {notice} days written notice.
5. JURISDICTION: Governed by laws of {juris}.
6. INDEMNIFICATION: Each party shall indemnify the other against third-party claims.

IN WITNESS WHEREOF the parties execute this Agreement on the date above.""",

        f"""NON-DISCLOSURE AGREEMENT (NDA)
Effective Date: {date}
Between: {party_a} and {party_b}

Whereas both parties wish to explore a potential business relationship,
each party (the "Disclosing Party") may share proprietary information.

OBLIGATIONS:
- Receiving Party shall not disclose Confidential Information to third parties.
- Obligations survive termination for a period of 3 years.
- Breach shall entitle Disclosing Party to seek injunctive relief.

JURISDICTION & GOVERNING LAW:
This Agreement is governed by the laws of {juris}.

AGREED AND EXECUTED on {date}.""",

        f"""MASTER SERVICE AGREEMENT
Parties: {party_a} ("Vendor") and {party_b} ("Buyer")    Date: {date}

1. SCOPE — Vendor agrees to provide services as described in each Statement of Work.
2. PAYMENT — Buyer shall remit payment of {amount} per engagement within 30 days.
3. WARRANTY — Vendor warrants services meet industry standards.
4. LIABILITY — Neither party shall be liable for indirect or consequential damages.
5. ARBITRATION — Disputes resolved via binding arbitration in {juris}.
6. ENTIRE AGREEMENT — This document constitutes the entire agreement between parties.

Signed on behalf of both parties with full authority.""",
    ]
    return random.choice(templates)


def _gen_affidavit() -> str:
    deponent = random.choice(PATIENT_NAMES)
    date     = random.choice(DATES)
    place    = random.choice(["Mumbai", "New York", "London", "Toronto", "Sydney"])

    templates = [
        f"""AFFIDAVIT
I, {deponent}, do hereby solemnly affirm and declare as follows:

1. I am a resident of {place} and am competent to make this affidavit.
2. The facts stated herein are true to the best of my knowledge and belief.
3. I have not suppressed any material fact that is within my knowledge.
4. This affidavit is made for the purpose of legal proceedings.

DEPONENT: {deponent}
Sworn and subscribed before me on this {date}.
Notary Public / Commissioner of Oaths.
Seal and Signature of Notary.""",

        f"""SWORN STATEMENT / AFFIDAVIT
Date: {date}    Place: {place}

I, {deponent}, being duly sworn on oath, hereby declare that:
- The information provided herein is accurate and complete.
- I have personal knowledge of the facts set forth in this affidavit.
- I make this declaration in support of pending legal proceedings.

I understand that false statements made herein are subject to legal penalty.

Signed: {deponent}
Before me: _________________________ (Notary)""",
    ]
    return random.choice(templates)


def _gen_invoice() -> str:
    vendor   = random.choice(COMPANIES)
    client   = random.choice([c for c in COMPANIES if c != vendor])
    date     = random.choice(DATES)
    inv_num  = f"INV-{random.randint(10000, 99999)}"
    items    = random.sample(INVOICE_ITEMS, random.randint(2, 4))
    subtotal = random.randint(5000, 50000)
    tax_rate = random.choice([0.05, 0.10, 0.18])
    tax      = round(subtotal * tax_rate)
    total    = subtotal + tax

    line_items = "\n".join(
        [f"  {item:<35} USD {random.randint(500,5000):,}" for item in items]
    )

    templates = [
        f"""TAX INVOICE
Invoice Number: {inv_num}    Invoice Date: {date}
From: {vendor}
Bill To: {client}

ITEMS:
{line_items}

                               Subtotal:  USD {subtotal:,}
                               Tax ({int(tax_rate*100)}%): USD {tax:,}
                               TOTAL DUE: USD {total:,}

Payment Due Date: 30 days from invoice date.
Bank Transfer / Cheque payable to {vendor}.""",

        f"""COMMERCIAL INVOICE — {inv_num}
Seller: {vendor}
Buyer:  {client}
Date:   {date}

Description of Services:
{line_items}

Sub-Total: USD {subtotal:,}
GST/Tax:   USD {tax:,}
Net Payable: USD {total:,}

Terms: Net 30. Late payment subject to 2% monthly interest.
Please reference invoice number in all correspondence.""",
    ]
    return random.choice(templates)


def _gen_unknown() -> str:
    snippets = [
        "The quick brown fox jumps over the lazy dog. Lorem ipsum dolor sit amet.",
        "Meeting notes from Monday. Action items: follow up, schedule review, confirm dates.",
        "Chapter 3: Introduction to Algorithms. This chapter covers sorting and searching.",
        "Dear Sir/Madam, I am writing to enquire about the vacancy posted online.",
        "Grocery list: milk, eggs, bread, butter, coffee, sugar, vegetables.",
        "Project update: Sprint 12 completed. Velocity: 34 points. Next sprint planning Friday.",
        "Thank you for your message. We will get back to you within 2 business days.",
    ]
    return random.choice(snippets)


# ─── Generator map ────────────────────────────────────────────────────────────
GENERATORS = {
    "medical_prescription": _gen_prescription,
    "lab_report":           _gen_lab_report,
    "legal_contract":       _gen_legal_contract,
    "affidavit":            _gen_affidavit,
    "invoice":              _gen_invoice,
    "unknown":              _gen_unknown,
}


# ─── Dataset builder ──────────────────────────────────────────────────────────

@dataclass
class Sample:
    text: str
    label: str
    label_id: int


def build_dataset(
    n_per_class: int = 200,
    val_split: float = 0.15,
    test_split: float = 0.10,
    output_dir: str = "data/samples",
) -> dict:
    """
    Build train/val/test splits and save as JSONL files.
    Returns dict with split sizes.

    Generates extra samples to compensate for duplicates and ensures
    no text overlap between train/val/test splits.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Generate unique samples per label, trying extra rounds if needed
    all_samples: List[Sample] = []
    for label, gen_fn in GENERATORS.items():
        seen_texts = set()
        max_attempts = n_per_class * 5  # avoid infinite loops
        attempts = 0
        while len(seen_texts) < n_per_class and attempts < max_attempts:
            text = gen_fn()
            if text not in seen_texts:
                seen_texts.add(text)
                all_samples.append(Sample(
                    text=text,
                    label=label,
                    label_id=LABEL2ID[label],
                ))
            attempts += 1

    random.shuffle(all_samples)
    n = len(all_samples)
    n_test = int(n * test_split)
    n_val  = int(n * val_split)
    n_train = n - n_test - n_val

    splits = {
        "train": all_samples[:n_train],
        "val":   all_samples[n_train:n_train + n_val],
        "test":  all_samples[n_train + n_val:],
    }

    for split_name, samples in splits.items():
        path = os.path.join(output_dir, f"{split_name}.jsonl")
        with open(path, "w") as f:
            for s in samples:
                f.write(json.dumps(asdict(s)) + "\n")
        print(f"[dataset_builder] Saved {len(samples)} samples → {path}")

    # Save label map
    with open(os.path.join(output_dir, "label_map.json"), "w") as f:
        json.dump({"label2id": LABEL2ID, "id2label": ID2LABEL}, f, indent=2)

    return {split: len(s) for split, s in splits.items()}


if __name__ == "__main__":
    stats = build_dataset(n_per_class=200)
    print(f"\nDataset built: {stats}")
    print(f"Total samples: {sum(stats.values())}")
    print(f"Labels: {list(LABEL2ID.keys())}")
