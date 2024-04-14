import csv
from jobspy import scrape_jobs
from pandas import DataFrame


def scrape_helper(title: str, offset: int, amount: int) -> DataFrame:
    jobs = scrape_jobs(
        site_name=["indeed", "linkedin", "zip_recruiter", "glassdoor"],
        search_term=title,
        location="United States",
        results_wanted=amount,
        offset=offset,
        hours_old=168,
        country_indeed='United States'
    )
    return jobs


HOT_JOBS_1 = [
    "Software Developer", "Marketing Manager", "Sales Associate", "Nurse", "Accountant",
    "Engineer", "Teacher", "Manager", "Graphic Designer", "Consultant",
    "Administrator", "Technician", "Data Analyst", "Registered Nurse", "Customer Service Representative"
]

HOT_JOBS_2 = [
    "Project Manager", "Product Manager",
    "Business Analyst", "Financial Analyst", "Human Resources Manager",
    "Pharmacist", "Physician", "Mechanic", "Electrician", "Plumber", "Lawyer"
]

OTHER_JOBS = ["Content Writer", "Editor", "IT Support Specialist", "Web Developer", "Network Engineer",
              "Chef", "Bartender", "Architect", "Civil Engineer", "Physical Therapist",
              "Paralegal", "UX/UI Designer", "Graphic Designer", "Public Relations Specialist",
              "Social Worker", "Police Officer", "Firefighter", "Account Executive", "Recruitment Specialist",
              "Real Estate Agent", "Insurance Agent", "Compliance Officer", "Event Planner", "Logistics Manager",
              "Warehouse Supervisor", "Quality Assurance Manager", "Safety Officer", "Environmental Scientist",
              "Lab Technician",
              "School Teacher", "College Professor", "Librarian", "Archaeologist",
              "Biomedical Engineer", "Chemical Engineer", "Systems Administrator", "Cybersecurity Specialist",
              "Network Administrator",
              "AI Engineer", "Cloud Architect", "Database Administrator", "Software Tester", "Mobile App Developer",
              "Art Director", "Fashion Designer", "Videographer", "Digital Marketing Specialist", "SEO Specialist",
              "Compliance Manager", "Brand Manager", "Advertising Coordinator", "Social Media Manager",
              "Game Developer", "Animation Artist", "3D Modeler", "Content Strategist",
              "Supply Chain Coordinator", "Forklift Operator", "Inventory Specialist",
              "Actuary", "Pilot", "Flight Attendant", "Tour Guide", "Travel Agent",
              "Fitness Instructor", "Nutritionist", "Dietitian",
              "Dental Assistant", "Veterinary Technician", "Research Coordinator", "Research Scientist",
              "Security Officer", "Private Investigator", "Lobbyist"]
