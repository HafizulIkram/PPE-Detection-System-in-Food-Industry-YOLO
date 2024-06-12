import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-GUI environments

from model import DetectionEvent
from datetime import timedelta
import pytz
import matplotlib.pyplot as plt
from fpdf import FPDF
import os
import tempfile

def generate_report(start_date, end_date):
    # Ensure the dates are timezone-aware
    start_date = pytz.utc.localize(start_date)
    end_date = pytz.utc.localize(end_date) + timedelta(days=1)  # Ensure the end date includes the entire day

    # Query the database for detection events within the specified date range
    events = DetectionEvent.query.filter(
        DetectionEvent.detection_time >= start_date,
        DetectionEvent.detection_time < end_date
    ).all()

    # Count the number of each class detected
    class_counts = {}
    for event in events:
        class_name = event.detected_class.name
        if class_name in class_counts:
            class_counts[class_name] += 1
        else:
            class_counts[class_name] = 1

    # Count the number of completion statuses
    status_counts = {}
    for event in events:
        status = event.completion_status.status
        if status in status_counts:
            status_counts[status] += 1
        else:
            status_counts[status] = 1

    # Generate the analysis paragraph
    class_chart_path, class_percentages = create_bar_chart(class_counts, "Class Distribution")
    status_chart_path, status_percentages = create_pie_chart(status_counts, "Completion Status Distribution")
    analysis_paragraph = generate_analysis_paragraph(class_percentages, status_percentages, start_date, end_date)

    return class_counts, status_counts, analysis_paragraph

def create_bar_chart(data, title):
    if not data:
        return None, {}

    labels = list(data.keys())
    values = list(data.values())
    total = sum(values)
    percentages = {label: (value / total) * 100 for label, value in data.items()}

    plt.figure(figsize=(10, 6))
    plt.bar(labels, values, color='blue')
    plt.xlabel('Classes')
    plt.ylabel('Counts')
    plt.title(title)
    chart_path = os.path.join(tempfile.gettempdir(), f"{title.replace(' ', '_')}.png")
    plt.savefig(chart_path)
    plt.close()
    return chart_path, percentages

def create_pie_chart(data, title):
    if not data:
        return None, {}

    labels = list(data.keys())
    values = list(data.values())
    total = sum(values)
    percentages = {label: (value / total) * 100 for label, value in data.items()}

    plt.figure(figsize=(6, 6))
    plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    plt.title(title)
    chart_path = os.path.join(tempfile.gettempdir(), f"{title.replace(' ', '_')}.png")
    plt.savefig(chart_path)
    plt.close()
    return chart_path, percentages

def generate_analysis_paragraph(class_percentages, status_percentages, start_date, end_date):
    analysis = f"In the period from {start_date.strftime('%d %B %Y')} to {end_date.strftime('%d %B %Y')}:\n\n"
    
    # Class Distribution Analysis
    analysis += "Based on the graph, the class distribution is as follows:\n"
    for class_label, percentage in class_percentages.items():
        analysis += f"The class {class_label} was detected around {percentage:.2f}%. "
    max_class = max(class_percentages, key=class_percentages.get)
    min_class = min(class_percentages, key=class_percentages.get)
    analysis += f"The most detected class is {max_class} with {class_percentages[max_class]:.2f}%, "
    analysis += f"and the least detected class is {min_class} with {class_percentages[min_class]:.2f}%.\n\n"

    # Completion Status Distribution Analysis
    analysis += "Regarding the completion status, the distribution is as follows:\n"
    for status_label, percentage in status_percentages.items():
        analysis += f"The status {status_label} was observed around {percentage:.2f}%. "
    max_status = max(status_percentages, key=status_percentages.get)
    min_status = min(status_percentages, key=status_percentages.get)
    analysis += f"The most observed status is {max_status} with {status_percentages[max_status]:.2f}%, "
    analysis += f"and the least observed status is {min_status} with {status_percentages[min_status]:.2f}%.\n"

    return analysis

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Report', 0, 1, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(5)

    def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, body)
        self.ln()

    def add_image(self, image_path):
        self.image(image_path, x=10, w=180)
        self.ln(5)

def create_pdf_report(class_counts, status_counts, analysis_paragraph, start_date, end_date, file_path):
    class_chart_path, class_percentages = create_bar_chart(class_counts, "Class Distribution")
    status_chart_path, status_percentages = create_pie_chart(status_counts, "Completion Status Distribution")

    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Report from {start_date.strftime('%d.%m.%Y')} to {end_date.strftime('%d.%m.%Y')}", ln=True, align='C')
    pdf.ln(5)
    
    # Add class bar chart and analysis
    pdf.chapter_title("Class Distribution")
    if class_chart_path:
        pdf.add_image(class_chart_path)
    pdf.chapter_body(analysis_paragraph.split('\n\n')[1])
    pdf.add_page()
    
    # Add status pie chart and analysis
    pdf.chapter_title("Completion Status Distribution")
    if status_chart_path:
        pdf.add_image(status_chart_path)
    pdf.chapter_body(analysis_paragraph.split('\n\n')[2])

    pdf.output(file_path)