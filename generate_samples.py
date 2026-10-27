"""
Generates 4 realistic sample enterprise files (PDF, TXT, CSV, Excel) inside sample_data/
so the evaluator can test multi-format ingestion immediately.
"""
import os
import pandas as pd
from fpdf import FPDF


def create_sample_files(output_dir: str = "sample_data"):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Sample PDF: Enterprise Cloud Security Policy
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf_lines = [
        "CLOUDGUARD ENTERPRISE SECURITY & COMPLIANCE POLICY (v2026.4)",
        "",
        "1. Encryption Standards:",
        "All production storage buckets in AWS S3, Azure Blob, and GCP Cloud Storage must enforce",
        "AES-256 encryption at rest and TLS 1.3 in transit. Unencrypted buckets must be remediated",
        "within 4 hours of detection.",
        "",
        "2. Identity and Access Management (IAM):",
        "Root accounts must have hardware MFA enabled. Privileged role sessions are restricted to",
        "a maximum duration of 1 hour. Service account keys must rotate every 90 days.",
        "",
        "3. FinOps Governance:",
        "Any cloud resource exceeding $5,000 monthly spend without an 'Environment' and 'CostCenter'",
        "tag will be flagged for executive audit review.",
    ]
    for line in pdf_lines:
        pdf.cell(200, 8, text=line, new_x="LMARGIN", new_y="NEXT")
    pdf.output(os.path.join(output_dir, "cloud_security_policy.pdf"))

    # 2. Sample TXT: DevSecOps Incident Log
    txt_content = (
        "INCIDENT LOG: INC-2026-0892\n"
        "Date: 2026-10-12 | Severity: HIGH | CSP: AWS (eu-west-2)\n"
        "Summary: Public read ACL detected on bucket 'cg-prod-audit-logs-2026'.\n"
        "Root Cause: Terraform pipeline bypassed OPA policy check during hotfix deployment.\n"
        "Resolution: Automated remediation lambda revoked public ACL in 14 minutes. "
        "Lead Auditor recommended mandatory pre-commit Checkov scanning on all repositories.\n"
    )
    with open(
        os.path.join(output_dir, "devsecops_incident_log.txt"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(txt_content)

    # 3. Sample CSV: Multi-Cloud FinOps Billing Export
    csv_data = pd.DataFrame(
        [
            {
                "Service": "AWS EC2",
                "Cloud": "AWS",
                "Department": "Core-Platform",
                "Monthly_Cost_USD": 8450,
                "Tagged": "No",
                "Anomaly_Detected": "Yes",
            },
            {
                "Service": "Azure Kubernetes Service",
                "Cloud": "Azure",
                "Department": "AI-ML-Research",
                "Monthly_Cost_USD": 6200,
                "Tagged": "Yes",
                "Anomaly_Detected": "No",
            },
            {
                "Service": "GCP BigQuery",
                "Cloud": "GCP",
                "Department": "Data-Analytics",
                "Monthly_Cost_USD": 3100,
                "Tagged": "Yes",
                "Anomaly_Detected": "No",
            },
            {
                "Service": "Oracle Autonomous DB",
                "Cloud": "OCI",
                "Department": "Finance-ERP",
                "Monthly_Cost_USD": 7900,
                "Tagged": "No",
                "Anomaly_Detected": "Yes",
            },
        ]
    )
    csv_data.to_csv(
        os.path.join(output_dir, "finops_billing_export.csv"), index=False
    )

    # 4. Sample Excel: CIS & ISO Compliance Matrix
    excel_data = pd.DataFrame(
        [
            {
                "Control_ID": "CIS-2.1.1",
                "Domain": "Storage Security",
                "Cloud_Provider": "AWS",
                "Status": "Non-Compliant",
                "Owner": "SecOps Team",
                "SLA_Hours": 4,
            },
            {
                "Control_ID": "ISO-A.9.4",
                "Domain": "Access Control (MFA)",
                "Cloud_Provider": "Azure",
                "Status": "Compliant",
                "Owner": "IAM Team",
                "SLA_Hours": 24,
            },
            {
                "Control_ID": "FIN-1.0.4",
                "Domain": "Untagged Spend > $5k",
                "Cloud_Provider": "OCI",
                "Status": "Non-Compliant",
                "Owner": "FinOps Lead",
                "SLA_Hours": 48,
            },
        ]
    )
    excel_data.to_excel(
        os.path.join(output_dir, "cis_compliance_matrix.xlsx"), index=False
    )

    return output_dir


if __name__ == "__main__":
    path = create_sample_files()
    print(f"Sample enterprise documents generated in: {path}/")
