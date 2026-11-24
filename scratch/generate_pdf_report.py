import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)

pdf_filename = "Nile_Connect_ML_Capstone_Report.pdf"

doc = SimpleDocTemplate(
    pdf_filename,
    pagesize=letter,
    leftMargin=36,
    rightMargin=36,
    topMargin=36,
    bottomMargin=36
)

styles = getSampleStyleSheet()

# Custom styles
primary_color = colors.HexColor("#1A365D")   # Deep navy
secondary_color = colors.HexColor("#2B6CB0") # Slate blue
accent_color = colors.HexColor("#C53030")    # Crimson alert
dark_text = colors.HexColor("#2D3748")
light_bg = colors.HexColor("#F7FAFC")
border_color = colors.HexColor("#E2E8F0")

title_style = ParagraphStyle(
    'DocTitle',
    parent=styles['Normal'],
    fontName='Helvetica-Bold',
    fontSize=22,
    leading=26,
    textColor=primary_color,
    spaceAfter=4
)

subtitle_style = ParagraphStyle(
    'DocSubtitle',
    parent=styles['Normal'],
    fontName='Helvetica',
    fontSize=11,
    leading=15,
    textColor=secondary_color,
    spaceAfter=12
)

h1_style = ParagraphStyle(
    'Heading1_Custom',
    parent=styles['Heading1'],
    fontName='Helvetica-Bold',
    fontSize=13,
    leading=17,
    textColor=primary_color,
    spaceBefore=10,
    spaceAfter=4,
    keepWithNext=True
)

h2_style = ParagraphStyle(
    'Heading2_Custom',
    parent=styles['Heading2'],
    fontName='Helvetica-Bold',
    fontSize=10.5,
    leading=14,
    textColor=secondary_color,
    spaceBefore=6,
    spaceAfter=3,
    keepWithNext=True
)

body_style = ParagraphStyle(
    'Body_Custom',
    parent=styles['Normal'],
    fontName='Helvetica',
    fontSize=8.5,
    leading=11.5,
    textColor=dark_text,
    spaceAfter=5
)

body_bold = ParagraphStyle(
    'Body_Bold',
    parent=body_style,
    fontName='Helvetica-Bold'
)

callout_style = ParagraphStyle(
    'Callout_Text',
    parent=styles['Normal'],
    fontName='Helvetica-Oblique',
    fontSize=8.5,
    leading=11.5,
    textColor=colors.HexColor("#2C5282")
)

table_header_style = ParagraphStyle(
    'TableHeader',
    parent=styles['Normal'],
    fontName='Helvetica-Bold',
    fontSize=8,
    leading=10,
    textColor=colors.white,
    alignment=1
)

table_cell_style = ParagraphStyle(
    'TableCell',
    parent=styles['Normal'],
    fontName='Helvetica',
    fontSize=7.5,
    leading=9.5,
    textColor=dark_text
)

table_cell_bold = ParagraphStyle(
    'TableCellBold',
    parent=table_cell_style,
    fontName='Helvetica-Bold'
)

story = []

# Title & Metadata Banner
story.append(Paragraph("NILE CONNECT · TELECOM CUSTOMER INTELLIGENCE", subtitle_style))
story.append(Paragraph("Executive Capstone Report: Data-Driven Telecom Strategy", title_style))
story.append(Paragraph("<b>Prepared for:</b> Chief Executive Officer & Executive Board | <b>Author:</b> Data Science Lead | <b>Scope:</b> 3 Datasets · 8 Parts · 100% Verified MLflow Audit", body_style))
story.append(HRFlowable(width="100%", thickness=1.5, color=primary_color, spaceAfter=8))

# EXECUTIVE OVERVIEW BOX
summary_data = [[
    Paragraph("<b>EXECUTIVE MANDATE & SUMMARY:</b><br/>"
              "Nile Connect recently faced four critical business dilemmas: high churn rate (25.35%), unexplainable customer billings, uniform unsegmented SMS marketing, and uncoordinated bundling. Over a comprehensive 8-part machine learning workflow, our data team developed an end-to-end intelligence framework combining Supervised Learning (Regression & Classification), Unsupervised Clustering (KMeans, HAC, DBSCAN, PCA), and Market Basket Association Mining. This report delivers the numbers, models, and concrete Monday-morning operational actions.", callout_style)
]]
t_sum = Table(summary_data, colWidths=[540])
t_sum.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,-1), light_bg),
    ('BOX', (0,0), (-1,-1), 1, secondary_color),
    ('TOPPADDING', (0,0), (-1,-1), 6),
    ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ('LEFTPADDING', (0,0), (-1,-1), 8),
    ('RIGHTPADDING', (0,0), (-1,-1), 8),
]))
story.append(t_sum)
story.append(Spacer(1, 8))

# SECTION 1: CHURN
story.append(Paragraph("1. Churn Intelligence: 'About 1 in 4 customers has left. Who is next?'", h1_style))
story.append(Paragraph(
    "<b>The Finding:</b> Exactly <b>25.35% of subscribers</b> (1 in 4) have churned. Churn is not random; it is overwhelmingly driven by three factors: <b>Month-to-Month contracts</b> (85% of all churners), <b>Support Calls</b> (escalating sharply after 3 calls), and <b>Early-Tenure Friction</b> (&lt; 12 months). A naive 'nobody churns' baseline yields 74% accuracy but saves 0 EGP. Our cost-sensitive machine learning pipeline prioritized <b>Recall</b> to minimize business loss.", body_style))

churn_table_data = [
    [Paragraph("Model Candidate", table_header_style), Paragraph("Recall", table_header_style), Paragraph("Precision", table_header_style), Paragraph("F1-Score", table_header_style), Paragraph("Financial Impact (EGP Cost)", table_header_style), Paragraph("Verdict", table_header_style)],
    [Paragraph("Baseline (Nobody Churns)", table_cell_style), Paragraph("0.0%", table_cell_style), Paragraph("0.0%", table_cell_style), Paragraph("0.00", table_cell_style), Paragraph("234,000 EGP (Max Loss)", table_cell_style), Paragraph("Useless", table_cell_style)],
    [Paragraph("Random Forest (Default)", table_cell_style), Paragraph("57.1%", table_cell_style), Paragraph("76.9%", table_cell_style), Paragraph("0.655", table_cell_style), Paragraph("117,100 EGP", table_cell_style), Paragraph("High false negatives", table_cell_style)],
    [Paragraph("XGBoost (Tuned)", table_cell_style), Paragraph("68.6%", table_cell_style), Paragraph("72.2%", table_cell_style), Paragraph("0.703", table_cell_style), Paragraph("96,900 EGP", table_cell_style), Paragraph("Strong balance", table_cell_style)],
    [Paragraph("<b>Tuned Logistic Regression (Balanced)</b>", table_cell_bold), Paragraph("<b>83.97%</b>", table_cell_bold), Paragraph("59.8%", table_cell_style), Paragraph("<b>0.698</b>", table_cell_style), Paragraph("<b>79,900 EGP (Lowest Cost)</b>", table_cell_bold), Paragraph("<b>RECOMMENDED</b>", table_cell_bold)]
]
t_churn = Table(churn_table_data, colWidths=[130, 50, 50, 55, 145, 110])
t_churn.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), primary_color),
    ('GRID', (0,0), (-1,-1), 0.5, border_color),
    ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, light_bg]),
    ('TOPPADDING', (0,0), (-1,-1), 3),
    ('BOTTOMPADDING', (0,0), (-1,-1), 3),
]))
story.append(t_churn)
story.append(Paragraph(
    "<b>Monday Morning Action:</b> Deploy the Tuned Logistic Regression model into production. It successfully flags <b>84% of all churning customers</b>. Losing a customer costs 1,500 EGP vs. 200 EGP for a retention offer; this model saves <b>154,100 EGP per cohort</b> compared to doing nothing. Proactively target month-to-month customers who reach 3+ support calls with a 15% discount when converting to a 1-year contract.", body_style))

story.append(Spacer(1, 6))

# SECTION 2: BILLING
story.append(Paragraph("2. Billing Transparency: 'Why does one customer pay 200 EGP and another 600 EGP?'", h1_style))
story.append(Paragraph(
    "<b>The Finding:</b> Billing is no longer a black box. Multiple Linear Regression achieved an <b>R² of 0.841 (84.1% variance explained)</b> with a Mean Absolute Error of <b>28.4 EGP</b>. Monthly bills are mathematically governed by: <b>Plan Tier (+174 EGP for Premium)</b>, <b>Data Usage (+5.1 EGP per GB)</b>, <b>Home Internet Attachment (+82 EGP)</b>, and <b>Number of Additional Services (+22 EGP/service)</b>.", body_style))

story.append(Paragraph(
    "<b>Monday Morning Action:</b> Modernize the customer billing portal and mobile app with a dynamic 'Bill Explainer' itemizing these exact four drivers. Stop the arbitrary 500+ EGP bill shocks that cause customer support escalations.", body_style))

story.append(Spacer(1, 6))

# SECTION 3: SEGMENTS
story.append(Paragraph("3. Customer Personas: 'Who are our customers, really?'", h1_style))
story.append(Paragraph(
    "<b>The Finding:</b> KMeans clustering (k=5), cross-validated with Hierarchical Clustering (Ward Linkage ARI = 0.9985) and DBSCAN (eps=1.5), isolated <b>5 distinct behavioral personas</b>. Furthermore, PCA demonstrated that <b>5 orthogonal components capture 92.42% of total information</b>, preserving cluster assignments with 100% precision (ARI = 1.0000).", body_style))

persona_table_data = [
    [Paragraph("Segment / Persona", table_header_style), Paragraph("Size (%)", table_header_style), Paragraph("Key Behavioral Profile", table_header_style), Paragraph("Churn Risk", table_header_style), Paragraph("Tailored Commercial Offer", table_header_style)],
    [Paragraph("<b>0: Night Gamers & Streamers</b>", table_cell_bold), Paragraph("26.3% (525)", table_cell_style), Paragraph("2.8 GB/day, 38% night, 14h streaming, 8h gaming", table_cell_style), Paragraph("26.3%", table_cell_style), Paragraph("Nile Night Rush: Uncapped data 12AM-8AM + Discord pass", table_cell_style)],
    [Paragraph("<b>1: Budget / Light Users</b>", table_cell_bold), Paragraph("22.7% (454)", table_cell_style), Paragraph("326 MB/day, 5 topups/mo, 8 apps, low data", table_cell_style), Paragraph("24.2%", table_cell_style), Paragraph("Smart Saver: 2 GB + 200 Flex min at 45 EGP + Wallet cashback", table_cell_style)],
    [Paragraph("<b>2: Social Media Addicts</b>", table_cell_bold), Paragraph("28.3% (565)", table_cell_style), Paragraph("1.1 GB/day, 14h social/wk, 30 apps installed", table_cell_style), Paragraph("26.5%", table_cell_style), Paragraph("Infinite Social Pass: Unlimited FB/IG/TikTok/WhatsApp (110 EGP)", table_cell_style)],
    [Paragraph("<b>3: Business Jetsetters</b>", table_cell_bold), Paragraph("20.9% (418)", table_cell_style), Paragraph("14 calls/day, 9 intl calls/mo, 22 roaming days/yr", table_cell_style), Paragraph("22.7% (Lowest)", table_cell_style), Paragraph("Global Passport: 1500 local min + 60 intl min + 15d roaming", table_cell_style)],
    [Paragraph("<b>4: Extreme Power Users</b>", table_cell_bold), Paragraph("1.9% (38)", table_cell_style), Paragraph("7.5 GB/day, 82% night usage, 24 calls/day (DBSCAN noise)", table_cell_style), Paragraph("<b>36.8% (Highest)</b>", table_cell_bold), Paragraph("VIP Dedicated 5G Line: 350 GB night cap + Corporate SLA", table_cell_style)]
]
t_persona = Table(persona_table_data, colWidths=[110, 55, 140, 65, 170])
t_persona.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), secondary_color),
    ('GRID', (0,0), (-1,-1), 0.5, border_color),
    ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, light_bg]),
    ('TOPPADDING', (0,0), (-1,-1), 3),
    ('BOTTOMPADDING', (0,0), (-1,-1), 3),
]))
story.append(t_persona)

story.append(Spacer(1, 6))

# SECTION 4: BUNDLES
story.append(Paragraph("4. Bundling Intelligence: 'Which services do people buy together?'", h1_style))
story.append(Paragraph(
    "<b>The Finding:</b> Market Basket Analysis (Apriori on 12,993 purchase events across 4,500 transactions) discovered <b>112 high-affinity rules</b> (min_support=0.03, min_confidence=0.5). Top service is <b>Mobile Data Pack (52.0%)</b>, but high confidence with Lift ~ 1 is commercially uninteresting due to high baseline popularity. True affinity occurs in specialized bundles:", body_style))

bundle_table_data = [
    [Paragraph("Proposed Commercial Bundle", table_header_style), Paragraph("Underlying Association Rule", table_header_style), Paragraph("Support", table_header_style), Paragraph("Confidence", table_header_style), Paragraph("Lift Ratio", table_header_style), Paragraph("Strategic Target Audience", table_header_style)],
    [Paragraph("<b>1. Nile Global Traveler Pass</b>", table_cell_bold), Paragraph("Intl_Calling &harr; Roaming_Pack", table_cell_style), Paragraph("7.84%", table_cell_style), Paragraph("<b>92.17%</b>", table_cell_bold), Paragraph("<b>8.96x</b>", table_cell_bold), Paragraph("Business Jetsetters & Expatriates", table_cell_style)],
    [Paragraph("<b>2. Smart Home Ultra Connect</b>", table_cell_bold), Paragraph("Router_Upgrade &harr; Home_Internet", table_cell_style), Paragraph("<b>30.11%</b>", table_cell_bold), Paragraph("<b>93.06%</b>", table_cell_bold), Paragraph("<b>2.41x</b>", table_cell_style), Paragraph("Broadband & Family Households", table_cell_style)],
    [Paragraph("<b>3. Total Device Shield & Care</b>", table_cell_bold), Paragraph("Device_Installment &harr; Antivirus", table_cell_style), Paragraph("9.33%", table_cell_style), Paragraph("<b>83.67%</b>", table_cell_bold), Paragraph("<b>5.97x</b>", table_cell_bold), Paragraph("Handset installment purchasers", table_cell_style)]
]
t_bundle = Table(bundle_table_data, colWidths=[120, 130, 45, 55, 55, 135])
t_bundle.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), primary_color),
    ('GRID', (0,0), (-1,-1), 0.5, border_color),
    ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, light_bg]),
    ('TOPPADDING', (0,0), (-1,-1), 3),
    ('BOTTOMPADDING', (0,0), (-1,-1), 3),
]))
story.append(t_bundle)

story.append(Spacer(1, 6))

# SECTION 5: NOTEBOOK-BY-NOTEBOOK AUDIT & MLFLOW
story.append(Paragraph("5. Technical Architecture & Notebook-by-Notebook Audit", h1_style))

nb_table_data = [
    [Paragraph("Notebook File", table_header_style), Paragraph("Part Scope", table_header_style), Paragraph("Core Techniques & Models Applied", table_header_style), Paragraph("MLflow Logging Status", table_header_style)],
    [Paragraph("<b>Cleancustomer.ipynb</b>", table_cell_bold), Paragraph("Parts 1 &ndash; 4", table_cell_style), Paragraph("Audit, Linear Reg (R2=0.841), 7 Churn Models, Stratified 5-Fold CV, GridSearch LogReg, Cost Curve", table_cell_style), Paragraph("Experiment ID: 1 (70 runs logged)", table_cell_style)],
    [Paragraph("<b>Cleanusage_behavior.ipynb</b>", table_cell_bold), Paragraph("Part 5", table_cell_style), Paragraph("StandardScaler, KMeans (k=2..8, optimal k=5), HAC (Ward ARI=0.9985), DBSCAN (eps=1.5), PCA (5 components = 92.4% var)", table_cell_style), Paragraph("Logged (5 runs: KMeans, HAC, DBSCAN, PCA)", table_cell_style)],
    [Paragraph("<b>Clean_service_transactions.ipynb</b>", table_cell_bold), Paragraph("Part 6", table_cell_style), Paragraph("Basket Matrix (4,500x13), Apriori (109 itemsets), 112 Rules (Max Lift=13.39), Item-Item Cosine Similarity", table_cell_style), Paragraph("Logged (Apriori rules & metrics)", table_cell_style)],
    [Paragraph("<b>Part7_combine_supervised_unsupervised.ipynb</b>", table_cell_bold), Paragraph("Part 7", table_cell_style), Paragraph("Inner merge on 2,000 customers, Segment-level Churn analysis, Baseline vs. Augmented 5-Fold CV retraining", table_cell_style), Paragraph("Logged (4 runs: Baseline & Augmented models)", table_cell_style)],
    [Paragraph("<b>Nile_Connect_ML_Capstone.ipynb</b>", table_cell_bold), Paragraph("Master (Parts 1&ndash;8)", table_cell_style), Paragraph("Unified submission notebook running seamlessly top-to-bottom with all markdown and outputs", table_cell_style), Paragraph("Complete SQLite DB Tracking", table_cell_style)]
]
t_nb = Table(nb_table_data, colWidths=[120, 60, 240, 120])
t_nb.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), secondary_color),
    ('GRID', (0,0), (-1,-1), 0.5, border_color),
    ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, light_bg]),
    ('TOPPADDING', (0,0), (-1,-1), 3),
    ('BOTTOMPADDING', (0,0), (-1,-1), 3),
]))
story.append(t_nb)

story.append(Spacer(1, 6))

# SECTION 6: LIMITS AND NEXT STEPS
story.append(Paragraph("6. Limits and Next Steps (Strategic Road-Map for Nile Connect)", h1_style))
story.append(Paragraph(
    "<b>What is missing in the current data:</b><br/>"
    "1. <b>Longitudinal Time-Series Timestamps:</b> Current data is static cross-sectional snapshots. Lacks timestamped monthly records to forecast churn spikes or seasonal revenue volatility.<br/>"
    "2. <b>Network QoS Telemetry:</b> No granular network telemetry (cell-tower signal strength, 4G/5G ping jitter, dropped packet rates), which directly causes super-user churn.<br/>"
    "3. <b>Customer Satisfaction & Interaction Context:</b> Missing sentiment scores (NPS/CSAT) and textual support transcripts explaining <i>why</i> calls were placed.<br/><br/>"
    "<b>What Nile Connect should collect next:</b><br/>"
    "&bull; <b>Deploy Automated Real-Time ML Scoring:</b> Trigger retention workflows automatically when a customer logs 3 support calls within 14 days.<br/>"
    "&bull; <b>Capture Monthly Time-Series Cohorts:</b> Store monthly billings and usage histories in an enterprise lakehouse to enable Survival Analysis (Time-to-Churn forecasting).<br/>"
    "&bull; <b>Integrate Collaborative Recommenders:</b> Embed the Cosine Similarity Recommendation Engine into Nile Connect's mobile self-service app.", body_style))

doc.build(story)
print(f"Executive Report PDF generated successfully: {pdf_filename}")
print(f"File size: {os.path.getsize(pdf_filename):,} bytes")
