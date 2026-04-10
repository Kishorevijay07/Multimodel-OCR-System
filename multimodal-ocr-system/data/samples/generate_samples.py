"""
Generate synthetic sample documents for testing.
Creates realistic medical and legal text samples as images.
"""

from PIL import Image, ImageDraw, ImageFont
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__))


def create_text_image(text: str, filename: str, size=(800, 1000)):
    """Render text onto a white image to simulate a scanned document."""
    img = Image.new("RGB", size, color="white")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        font_bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except:
        font = ImageFont.load_default()
        font_bold = font

    y = 40
    for line in text.strip().split("\n"):
        is_header = line.strip().isupper() or line.startswith("##")
        f = font_bold if is_header else font
        draw.text((40, y), line.strip("# "), fill="black", font=f)
        y += 28

    path = os.path.join(OUTPUT_DIR, filename)
    img.save(path)
    print(f"Saved: {path}")
    return path


MEDICAL_PRESCRIPTION = """
## CITY GENERAL HOSPITAL
## MEDICAL PRESCRIPTION

Patient Name: John Doe
Age: 45 Years         Gender: Male
Date: 14/03/2024

Doctor: Dr. Sarah Williams, MBBS, MD
License No: MH-2024-78901

Rx:

1. Amoxicillin 500mg
   Sig: 1 Capsule twice daily for 7 days
   Dispense: 14 Capsules

2. Ibuprofen 400mg
   Sig: 1 Tablet thrice daily after meals
   Dispense: 21 Tablets

3. Omeprazole 20mg
   Sig: 1 Capsule once daily before breakfast
   Dispense: 7 Capsules

Diagnosis: Acute Upper Respiratory Tract Infection
Allergies: None known

Refills: 0
Signature: Dr. Sarah Williams
"""

LAB_REPORT = """
## PATHCARE DIAGNOSTICS LAB
## LABORATORY TEST REPORT

Patient: Jane Smith          Age: 38 Years
Sample ID: LAB-2024-00421    Date: 15/03/2024
Referred By: Dr. A. Kumar

## COMPLETE BLOOD COUNT (CBC)

Test             Result       Reference Range   Status
Hemoglobin       11.2 g/dL   12.0-16.0 g/dL   LOW
WBC              7.5 K/uL    4.5-11.0 K/uL    NORMAL
RBC              3.9 M/uL    3.9-5.2 M/uL     NORMAL
Platelet         180 K/uL    150-400 K/uL      NORMAL

## BLOOD GLUCOSE

Fasting Glucose: 108 mg/dL    (Normal: < 100 mg/dL)  BORDERLINE
HbA1c:          6.1 %         (Normal: < 5.7 %)       ELEVATED

Comments: Patient shows borderline fasting glucose.
Follow-up advised in 3 months.

Verified By: Dr. P. Mehta, MD Pathology
"""

LEGAL_CONTRACT = """
## SERVICE AGREEMENT CONTRACT

This Agreement is entered into on this 10th day of March, 2024,
between:

XYZ Technology Solutions Pvt. Ltd., a company incorporated under
the Companies Act (hereinafter referred to as "Service Provider")

AND

ABC Retail Corporation, a company incorporated under the Companies
Act (hereinafter referred to as "Client")

## 1. SCOPE OF SERVICES

The Service Provider shall provide software development and
IT consulting services as mutually agreed.

## 2. PAYMENT TERMS

The Client agrees to pay USD 15,000 per month for services rendered.
Payment shall be made within 30 days of invoice receipt.

## 3. CONFIDENTIALITY

Both parties shall maintain strict confidentiality of all
proprietary information shared during this engagement.

## 4. TERMINATION

Either party may terminate this agreement with 30 days written notice.

## 5. JURISDICTION

This Agreement shall be governed by the laws of the State of
California, United States of America.

IN WITNESS WHEREOF, the parties have executed this Agreement.

XYZ Technology Solutions Pvt. Ltd.     ABC Retail Corporation
Authorized Signatory                   Authorized Signatory
"""


if __name__ == "__main__":
    create_text_image(MEDICAL_PRESCRIPTION, "sample_prescription.png")
    create_text_image(LAB_REPORT, "sample_lab_report.png")
    create_text_image(LEGAL_CONTRACT, "sample_legal_contract.png")
    print("\nAll sample documents generated!")