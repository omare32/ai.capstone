#!/usr/bin/env python
"""
PDF Report Generator for AAVAIL Revenue Prediction
Converts HTML report to professional PDF format
"""

import os
import sys
import webbrowser
from pathlib import Path

def generate_pdf_instructions():
    """
    Generate instructions for PDF creation since automated PDF generation
    requires additional dependencies that might not be available
    """
    
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    html_file = project_root / "reports" / "Part1_Data_Investigation_Report.html"
    
    print("=" * 70)
    print("AAVAIL Revenue Prediction - PDF Report Generator")
    print("=" * 70)
    
    if not html_file.exists():
        print("❌ HTML report not found!")
        print(f"Expected location: {html_file}")
        return False
    
    print("✅ HTML report found successfully!")
    print(f"Location: {html_file}")
    
    print("\n📄 PDF Generation Options:")
    print("\n1. BROWSER METHOD (Recommended):")
    print("   - The HTML report will open in your default browser")
    print("   - Press Ctrl+P (Windows) or Cmd+P (Mac)")
    print("   - Select 'Save as PDF' as destination")
    print("   - Choose 'More settings' and select:")
    print("     • Paper size: A4")
    print("     • Margins: Default")
    print("     • Scale: 100%")
    print("     • Background graphics: ✓ (checked)")
    print("   - Click 'Save' and choose your location")
    
    print("\n2. AUTOMATED METHOD (Advanced):")
    print("   Install additional dependencies:")
    print("   pip install weasyprint")
    print("   # or")
    print("   pip install pdfkit")
    print("   # Then run the automated_pdf_generation() function")
    
    # Open HTML file in browser
    try:
        webbrowser.open(f"file://{html_file.absolute()}")
        print(f"\n🌐 Opening HTML report in your default browser...")
        print("   You can now print to PDF using your browser's print function")
        return True
    except Exception as e:
        print(f"\n❌ Could not open browser automatically: {e}")
        print(f"   Please manually open: {html_file}")
        return False

def automated_pdf_generation():
    """
    Automated PDF generation (requires additional dependencies)
    """
    try:
        import weasyprint
        
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        html_file = project_root / "reports" / "Part1_Data_Investigation_Report.html"
        pdf_file = project_root / "reports" / "Part1_Data_Investigation_Report.pdf"
        
        print("🔄 Generating PDF using WeasyPrint...")
        
        # Generate PDF
        weasyprint.HTML(filename=str(html_file)).write_pdf(str(pdf_file))
        
        print(f"✅ PDF generated successfully!")
        print(f"Location: {pdf_file}")
        return True
        
    except ImportError:
        print("❌ weasyprint not installed. Install with: pip install weasyprint")
        return False
    except Exception as e:
        print(f"❌ Error generating PDF: {e}")
        return False

def alternative_pdf_generation():
    """
    Alternative PDF generation using pdfkit
    """
    try:
        import pdfkit
        
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        html_file = project_root / "reports" / "Part1_Data_Investigation_Report.html"
        pdf_file = project_root / "reports" / "Part1_Data_Investigation_Report.pdf"
        
        print("🔄 Generating PDF using pdfkit...")
        
        # PDF generation options for better formatting
        options = {
            'page-size': 'A4',
            'margin-top': '0.75in',
            'margin-right': '0.75in',
            'margin-bottom': '0.75in',
            'margin-left': '0.75in',
            'encoding': "UTF-8",
            'no-outline': None,
            'enable-local-file-access': None,
            'print-media-type': None
        }
        
        pdfkit.from_file(str(html_file), str(pdf_file), options=options)
        
        print(f"✅ PDF generated successfully!")
        print(f"Location: {pdf_file}")
        return True
        
    except ImportError:
        print("❌ pdfkit not installed. Install with: pip install pdfkit")
        print("   Note: pdfkit also requires wkhtmltopdf to be installed")
        return False
    except Exception as e:
        print(f"❌ Error generating PDF: {e}")
        return False

def main():
    """Main function to handle PDF generation"""
    
    print("Choose PDF generation method:")
    print("1. Browser method (opens HTML in browser for manual PDF save)")
    print("2. Automated method (requires weasyprint)")
    print("3. Alternative automated method (requires pdfkit)")
    
    try:
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            generate_pdf_instructions()
        elif choice == "2":
            if not automated_pdf_generation():
                print("\nFalling back to browser method...")
                generate_pdf_instructions()
        elif choice == "3":
            if not alternative_pdf_generation():
                print("\nFalling back to browser method...")
                generate_pdf_instructions()
        else:
            print("Invalid choice. Using browser method...")
            generate_pdf_instructions()
            
    except KeyboardInterrupt:
        print("\n\n👋 PDF generation cancelled by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Falling back to browser method...")
        generate_pdf_instructions()

if __name__ == "__main__":
    main()
