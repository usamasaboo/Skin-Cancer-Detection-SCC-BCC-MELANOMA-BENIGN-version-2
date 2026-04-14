from fpdf import FPDF
import os
from datetime import datetime

class CancerReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Skin Cancer Detection Analysis Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_pdf_report(predictions, original_img_path, heatmap_img_path, output_path):
    pdf = CancerReport()
    pdf.add_page()
    
    # Report Info
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, f'Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1)
    pdf.ln(5)

    # Risk Classification
    top_pred = predictions[0]['class'].lower()
    if top_pred == 'melanoma':
        risk_level = "HIGH RISK"
        risk_color = (190, 18, 60) # Danger Red
    elif top_pred in ['bcc', 'scc']:
        risk_level = "MODERATE RISK"
        risk_color = (161, 98, 7) # Warning Amber
    else:
        risk_level = "LOW RISK"
        risk_color = (21, 128, 61) # Success Green

    pdf.set_font('Arial', 'B', 14)
    pdf.set_text_color(*risk_color)
    pdf.cell(0, 10, f"Clinical Risk Level: {risk_level}", 0, 1)
    pdf.set_text_color(0, 0, 0) # Reset
    pdf.ln(5)

    # Analysis Results
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Diagnostic Confidence Breakdown:', 0, 1)
    pdf.ln(2)
    
    # Table Header
    pdf.set_font('Arial', 'B', 12)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(90, 10, 'Classification Type', 1, 0, 'C', 1)
    pdf.cell(90, 10, 'Confidence Score', 1, 1, 'C', 1)
    
    # Table Body
    pdf.set_font('Arial', '', 12)
    for pred in predictions:
        class_name = pred['class'].upper()
        prob_val = f"{pred['probability']:.2f}%"
        
        # Highlight top result
        if pred == predictions[0]:
            pdf.set_font('Arial', 'B', 12)
            pdf.set_text_color(37, 99, 235) # Accent Blue
        else:
            pdf.set_font('Arial', '', 12)
            pdf.set_text_color(0, 0, 0)
            
        pdf.cell(90, 10, f"  {class_name}", 1, 0, 'L')
        pdf.cell(90, 10, f"{prob_val}  ", 1, 1, 'R')
    
    pdf.set_text_color(0, 0, 0) # Reset color
    
    pdf.ln(10)

    # Images
    col_width = 80
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(col_width, 10, 'Original Image', 0, 0, 'C')
    pdf.cell(col_width, 10, 'Explainability Map', 0, 1, 'C')
    
    y_start = pdf.get_y()
    pdf.image(original_img_path, x=10, y=y_start, w=75)
    pdf.image(heatmap_img_path, x=100, y=y_start, w=75)
    
    pdf.ln(80) # Move down after images

    # Disclaimer
    pdf.ln(10)
    pdf.set_font('Arial', 'I', 10)
    disclaimer = "This AI system is for assistive diagnosis only and not a substitute for professional medical advice. Please consult a qualified healthcare professional for medical advice."
    pdf.multi_cell(0, 10, disclaimer)

    pdf.output(output_path)
    return output_path


def generate_comprehensive_report(history, username, output_path):
    pdf = CancerReport()
    pdf.set_title('Clinical Diagnostic Portfolio')
    pdf.set_author('MEDVISION AI')
    pdf.add_page()
    
    # Title Section
    pdf.set_font('Arial', 'B', 20)
    pdf.set_text_color(29, 78, 216) # Primary Medical Blue
    pdf.cell(0, 20, 'Clinical Diagnostic Portfolio', 0, 1, 'L')
    
    pdf.set_font('Arial', '', 12)
    pdf.set_text_color(100, 116, 139) # Text Muted
    pdf.cell(0, 8, f'Patient Name: {username}', 0, 1)
    pdf.cell(0, 8, f'Compiled On: {datetime.now().strftime("%B %d, %Y")}', 0, 1)
    pdf.ln(10)
    
    # Summary of Findings
    pdf.set_font('Arial', 'B', 14)
    pdf.set_text_color(15, 23, 42) # Text Dark
    pdf.cell(0, 10, 'Executive Diagnostic Summary', 0, 1)
    pdf.ln(2)
    
    # Table Header
    pdf.set_font('Arial', 'B', 11)
    pdf.set_fill_color(248, 250, 252) # bg-surface
    pdf.set_draw_color(226, 232, 240) # border-color
    
    pdf.cell(45, 10, '  Date & Time', 1, 0, 'L', 1)
    pdf.cell(65, 10, '  Classification', 1, 0, 'L', 1)
    pdf.cell(40, 10, '  Confidence', 1, 0, 'C', 1)
    pdf.cell(40, 10, '  Status', 1, 1, 'C', 1)
    
    # Table Body
    pdf.set_font('Arial', '', 10)
    for item in history:
        # Check if we need a new page
        if pdf.get_y() > 250:
            pdf.add_page()
            
        pdf.set_text_color(100, 116, 139)
        pdf.cell(45, 10, f"  {item['timestamp'][:16]}", 1, 0, 'L')
        
        # Color classification based on severity
        is_malignant = any(x in item['prediction'] for x in ['Melanoma', 'Malignant', 'Carcinoma', 'scc', 'bcc', 'melanoma'])
        if is_malignant:
            pdf.set_text_color(190, 18, 60) # Danger Red
        else:
            pdf.set_text_color(21, 128, 61) # Success Green
            
        pdf.cell(65, 10, f"  {item['prediction'].upper()}", 1, 0, 'L')
        
        pdf.set_text_color(15, 23, 42)
        pdf.cell(40, 10, f"  {item['confidence']}  ", 1, 0, 'C')
        
        status = 'REVIEW REQ' if is_malignant else 'MONITOR'
        pdf.cell(40, 10, f"  {status}  ", 1, 1, 'C')

    pdf.ln(15)
    
    # Clinical Notes Section
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Clinical Protocol Notes', 0, 1)
    pdf.set_font('Arial', '', 10)
    pdf.set_text_color(100, 116, 139)
    notes = (
        "This portfolio contains all digitized diagnostic interactions between the patient and "
        "the MEDVISION AI system. Each classification result is a combination of multi-layer "
        "ensemble model analysis and clinical morphological refinement (ABCDE validation). "
        "Red-labeled entries indicate structural markers consistent with malignancy and warrant "
        "immediate dermatological verification."
    )
    pdf.multi_cell(0, 8, notes)

    # Final Verification
    pdf.ln(10)
    pdf.set_font('Arial', 'I', 8)
    disclaimer = (
        "HIPAA COMPLIANCE NOTICE: This document contains Protected Health Information (PHI). "
        "Authorized clinical use only. MEDVISION AI is a secondary diagnostic aide and does not "
        "replace board-certified dermatological assessment."
    )
    pdf.multi_cell(0, 5, disclaimer)

    pdf.output(output_path)
    return output_path
