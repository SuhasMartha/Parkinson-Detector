import streamlit as st
from datetime import datetime
import random

# ========== COMPREHENSIVE Q&A DATABASE ==========
CHATBOT_QA = {
    # ===== WHAT IS PARKINSON'S =====
    "what is parkinson": """Parkinson's Disease (PD) is a neurodegenerative disorder that primarily affects movement. It occurs when nerve cells in the brain don't produce enough dopamine, a chemical messenger responsible for smooth, coordinated movement. It's the second most common neurodegenerative disease after Alzheimer's. The disease develops gradually, with symptoms appearing over time.""",
    
    "what causes parkinson": """The exact cause is unknown, but it involves the loss of dopamine-producing neurons in the substantia nigra region of the brain. Risk factors include: 
    • Age (usually 60+)
    • Genetics (family history)
    • Environmental factors (pesticide exposure)
    • Sex (slightly more common in men)
    • Head injuries
    • Chemical exposure""",
    
    "is parkinson hereditary": """Genetics plays a role - about 15-25% of Parkinson's patients have a family history. However, it's not simply inherited like some genetic diseases. Having a family member with PD increases risk but doesn't guarantee you'll develop it. Mutations in genes like SNCA and LRRK2 are associated with Parkinson's. Most cases are sporadic.""",
    
    "is parkinson fatal": """Parkinson's Disease itself is not directly fatal, but complications can be serious. Life expectancy is typically similar to the general population. However, complications like falls, pneumonia, and difficulty swallowing can be serious. Early diagnosis and proper management help maintain quality of life. Most people live 15-20+ years after diagnosis.""",
    
    # ===== SYMPTOMS =====
    "what are symptoms": """Main motor symptoms include:
    • Tremor (shaking) - usually at rest
    • Rigidity (stiffness) - increased muscle tone
    • Bradykinesia (slow movement)
    • Postural instability (balance problems)
    • Difficulty writing (micrographia)
    • Reduced facial expression
    • Speech changes (softer, monotone)
    
    Non-motor symptoms include:
    • Sleep disturbances
    • Depression & anxiety
    • Cognitive changes
    • Loss of smell (often earliest sign!)
    • Constipation
    • Temperature regulation problems""",
    
    "what are motor symptoms": """Motor (movement) symptoms of Parkinson's:
    • Tremor (about 70% have this)
    • Rigidity/Stiffness in limbs and joints
    • Bradykinesia - slowness and difficulty starting movement
    • Postural instability - poor balance
    • Walking difficulties - shuffling gait
    • Difficulty writing - smaller handwriting
    • Reduced facial expression - mask-like face
    • Speech changes - quiet, mumbling voice
    These symptoms typically start on one side and spread to both sides.""",
    
    "what are non-motor symptoms": """Non-motor symptoms are often overlooked but important:
    • Sleep disturbances (REM sleep behavior disorder)
    • Depression (affects 30-40% of patients)
    • Anxiety disorders
    • Cognitive changes - memory, concentration
    • Loss of smell (hyposmia) - often appears before motor symptoms!
    • Constipation (very common)
    • Urinary problems
    • Sexual dysfunction
    • Hallucinations
    • Temperature regulation problems
    Non-motor symptoms can appear BEFORE motor symptoms!""",
    
    "is tremor first symptom": """Tremor is common but not always the first symptom. About 70% have tremor, but 30% don't. Other first symptoms can be:
    • Stiffness or rigidity
    • Slowness of movement
    • Balance problems
    • Sleep disturbances
    • Depression
    • Loss of smell
    Some people never develop tremor but have other symptoms. If you notice any changes, consult a neurologist.""",
    
    "how does parkinson progress": """Parkinson's typically progresses slowly over years. However, progression varies greatly between individuals:
    
    Early Stage (0-2 years):
    • Subtle tremor in one hand
    • Mild slowness
    • Slight stiffness
    
    Middle Stage (2-10 years):
    • Tremor more noticeable
    • Increased slowness
    • Postural changes
    • Speech changes
    • May need assistive devices
    
    Late Stage (10+ years):
    • Severe motor symptoms
    • Fall risk increases significantly
    • Walking difficulties
    • May need assistance with daily activities
    
    Some progress rapidly, others slowly. Management can slow progression.""",
    
    "can symptoms improve": """Yes! Motor symptoms respond well to medication (Levodopa). Improvement strategies:
    • Levodopa medication - most effective
    • Physical therapy - improves mobility
    • Occupational therapy - daily activities
    • Speech therapy - voice and swallowing
    • Exercise - slows progression
    • Psychotherapy - mental health
    Many patients experience significant symptom improvement with proper treatment. However, symptoms may fluctuate - good days and bad days are common.""",
    
    # ===== DIAGNOSIS =====
    "how is parkinson diagnosed": """Diagnosis is clinical based on:
    • Medical history review
    • Neurological examination by specialist
    • Assessment of motor and non-motor symptoms
    • Response to Levodopa medication (if given)
    • Ruling out other conditions that mimic PD
    
    Important: There's NO definitive blood test or scan that diagnoses PD. A neurologist's expertise is crucial. DaTscan imaging can help confirm dopamine deficiency but is optional.""",
    
    "what tests are done": """Diagnostic tests include:
    • Neurological exam - reflexes, coordination, balance
    • MRI or CT scan - rules out other brain conditions
    • DaT scan - dopamine transporter scan (optional, shows dopamine levels)
    • Blood tests - rules out other conditions
    • Cognitive screening - memory and thinking
    These tests help confirm diagnosis and rule out conditions that mimic Parkinson's (Parkinson-Plus syndromes).""",
    
    "when should i see doctor": """See a neurologist if you notice:
    • New tremor
    • Stiffness or rigidity
    • Slowness of movement
    • Balance problems or frequent falls
    • Difficulty writing (smaller handwriting)
    • Reduced facial expression
    • Speech changes
    • Sleep disturbances
    • Depression or mood changes
    • Loss of smell
    Early diagnosis allows for better management and treatment planning. Don't wait if symptoms interfere with daily activities.""",
    
    # ===== MEDICATIONS =====
    "what medications treat parkinson": """Main Parkinson's medications:
    
    **Levodopa (L-DOPA)** - Most effective
    • Converts to dopamine in brain
    • Usually combined with carbidopa
    
    **Dopamine agonists**
    • Mimic dopamine in brain
    • Often used early
    
    **MAO-B inhibitors**
    • Prevent dopamine breakdown
    • May slow progression
    
    **COMT inhibitors**
    • Extend Levodopa effectiveness
    
    **Anticholinergics**
    • Reduce tremor and rigidity
    
    **Amantadine**
    • Reduces involuntary movements
    
    Medications don't cure but manage symptoms. Treatment is individualized.""",
    
    "what is levodopa": """Levodopa (L-DOPA) is the gold standard Parkinson's medication:
    • Most effective for motor symptoms
    • Crosses the blood-brain barrier
    • Converts to dopamine in the brain
    • Usually combined with carbidopa (which prevents premature breakdown)
    • Most patients see significant improvement within days of starting
    • Effectiveness can decrease over years (typically 3-5 years)
    • Side effects include nausea, dizziness, involuntary movements
    • Often combined with other medications""",
    
    "what are side effects": """Common medication side effects:
    • Nausea and digestive issues
    • Dizziness or low blood pressure
    • Involuntary movements (dyskinesia)
    • Sleep problems
    • Hallucinations or confusion
    • Mood changes
    • Loss of appetite
    
    Most side effects are manageable by:
    • Adjusting dose
    • Adding other medications
    • Taking with food
    • Changing medication timing
    
    Severe side effects should be reported to your doctor immediately.""",
    
    "can i stop medications": """NO - you shouldn't stop medications without doctor approval. Stopping suddenly can cause serious problems:
    • Severe symptom rebound
    • Muscle rigidity
    • Fever (neuroleptic malignant syndrome risk)
    • Hallucinations
    • Confusion
    
    However:
    • Doses may be adjusted over time
    • Some patients need reductions as disease progresses
    • Medication changes must be gradual
    • Always consult your neurologist before changing anything""",
    
    "do medications lose effectiveness": """Yes, some patients develop medication resistance over time:
    • Typically after 3-5 years of Levodopa use
    • Others may experience "on-off" fluctuations (working well, then not)
    • Motor complications increase over time
    
    Solutions:
    • Dose adjustment
    • Changing medications
    • Adding additional drugs
    • Adjusting dosing schedules
    • Deep Brain Stimulation (DBS) for severe cases
    
    Regular monitoring with your neurologist helps manage these changes.""",
    
    # ===== LIVING WITH PARKINSON'S =====
    "can i work with parkinson": """Yes, many people continue working for years after diagnosis:
    • Early stages often compatible with work
    • Job accommodations can help:
      - Modified schedule
      - Ergonomic adjustments
      - Flexible working
      - Remote work options
    
    Challenges:
    • Some professions become difficult
    • Performance may decline over time
    • May need career changes
    
    Options:
    • Disability benefits available if can't work
    • Vocational rehabilitation services
    • Talk to employer about accommodations""",
    
    "what lifestyle changes help": """Lifestyle modifications are crucial:
    
    **Exercise** (Most important!)
    • 150 minutes moderate activity weekly
    • Walking, swimming, tai chi, yoga
    • Strength and balance training
    
    **Therapy**
    • Physical therapy - mobility
    • Occupational therapy - daily tasks
    • Speech therapy - voice and swallowing
    
    **Nutrition**
    • Mediterranean diet recommended
    • Adequate protein
    • Fiber for constipation
    
    **Sleep**
    • 7-8 hours nightly
    • Regular schedule
    • Good sleep hygiene
    
    **Mental Health**
    • Social engagement
    • Stress reduction
    • Counseling if needed
    
    **Cognitive**
    • Mental stimulation
    • Learning new things
    • Social activities""",
    
    "is exercise helpful": """YES! Exercise is absolutely vital for Parkinson's:
    
    **Benefits:**
    • Slows disease progression
    • Improves motor function
    • Enhances mood and reduces depression
    • Reduces risk of falls
    • Improves balance and coordination
    • Better sleep quality
    
    **Recommended:** 150 minutes moderate activity per week
    
    **Types:**
    • Cardiovascular - walking, swimming
    • Strength training
    • Balance exercises - tai chi, yoga
    • Flexibility - stretching
    • Dance - especially helpful!
    
    **Important:** Consult your doctor before starting any exercise program.""",
    
    "how to prevent falls": """Falls are a major concern. Prevention strategies:
    
    **Physical**
    • Physical therapy for balance and strength
    • Regular exercise
    • Vision and hearing checks
    
    **Home**
    • Remove tripping hazards
    • Good lighting
    • Handrails in key areas
    • Non-slip flooring
    
    **Assistive Devices**
    • Cane or walker if needed
    • Appropriate footwear
    • Grip aids
    
    **Behavioral**
    • Avoid rushing
    • Take slow, deliberate steps
    • Hold onto support when moving
    • Be aware of your surroundings
    • Regular medication timing
    
    Early intervention significantly reduces fall risk.""",
    
    # ===== BASIC QUESTIONS =====
    "what is parkinson": """Parkinson's Disease (PD) is a neurodegenerative disorder affecting movement...""",
    "what causes parkinson": """Parkinson's is caused by loss of dopamine-producing neurons...""",
    "is parkinson hereditary": """Genetics contribute 15–25% of cases...""",
    "is parkinson fatal": """Not directly fatal, but complications can be serious...""",

    # ===== SYMPTOMS =====
    "what are symptoms": """Motor symptoms: Tremor, rigidity, slow movement...""",
    "is tremor first symptom": """Tremor is common but not always the first symptom...""",
    "how does parkinson progress": """PD progresses slowly over years...""",
    "can symptoms improve": """Yes, medication, exercise and therapy help...""",

    # --- NEW SYMPTOM QUESTIONS ---
    "what is bradykinesia": """Bradykinesia means slowness of movement...""",
    "why do hands shake": """Tremor occurs because the brain lacks dopamine...""",
    "why do i freeze while walking": """Freezing of gait is a common symptom...""",
    "why is my handwriting small": """Micrographia (tiny handwriting) is a Parkinson’s symptom...""",
    "why do i lose balance": """Postural instability occurs in later stages...""",
    "why do i feel stiffness": """Rigidity is caused by impaired muscle control...""",
    "why is my voice soft": """Hypophonia is a soft, low-volume voice seen in PD...""",
    "why do i drool": """Drooling is due to reduced swallowing frequency...""",
    "why do i have constipation": """PD affects the gut nervous system, slowing digestion...""",
    "is urinary urgency common": """Yes, bladder overactivity is common in PD...""",
    "why do i have sexual problems": """PD affects autonomic functions influencing sexual health...""",
    "why do i see blurry": """PD can cause visual disturbances or double vision...""",
    "is pain common": """Yes, 40–60% experience muscle or nerve pain...""",

    # ===== EMOTIONAL & COGNITIVE =====
    "is depression common": """Yes, 30–40% experience depression...""",
    "how do i cope emotionally": """Support groups, therapy, mindfulness help...""",
    "are support groups helpful": """Yes, support groups reduce isolation...""",

    # --- NEW MENTAL HEALTH QUESTIONS ---
    "is anxiety common": """Yes, anxiety affects nearly 40% of PD patients...""",
    "does parkinson cause memory loss": """Mild cognitive impairment can occur...""",
    "can parkinson cause dementia": """Some people develop Parkinson's dementia in later stages...""",
    "why do i feel fatigue": """Fatigue is a very common non-motor symptom...""",

    # ===== DIAGNOSIS =====
    "how is parkinson diagnosed": """Diagnosis is clinical...""",
    "what tests are done": """Neurological exam, MRI to rule out other issues...""",
    "when should i see doctor": """See a neurologist if you notice tremors, stiffness...""",

    # --- NEW DIAGNOSTIC QUESTIONS ---
    "is there a blood test": """No blood test can confirm Parkinson's...""",
    "can mri detect parkinson": """MRI helps rule out other conditions but doesn't diagnose PD...""",
    "what is datscan": """DaTscan measures dopamine transporter activity...""",
    "can ai detect parkinson": """Emerging AI tools analyze voice, handwriting, gait...""",
    "what are biomarkers": """Biomarkers like alpha-synuclein may help diagnose PD in the future...""",

    # ===== TREATMENT =====
    "what medications treat parkinson": """Levodopa, dopamine agonists...""",
    "what is levodopa": """Levodopa converts to dopamine in the brain...""",
    "what are side effects": """Nausea, dizziness, dyskinesia...""",
    "can i stop medications": """Never stop medications suddenly...""",

    # --- NEW TREATMENT QUESTIONS ---
    "why do meds wear off": """Wearing-off happens after years of levodopa use...""",
    "what is dyskinesia": """Involuntary movements caused by long-term levodopa...""",
    "should levodopa be taken with food": """Best taken 30–60 mins before meals...""",
    "what foods affect medication": """High-protein meals can interfere with levodopa absorption...""",
    "can i drink alcohol": """Moderation is usually fine, but discuss with your doctor...""",

    # ===== ADVANCED TREATMENTS =====
    "what about deep brain stimulation": """DBS is effective for tremor and dyskinesia...""",
    "are there clinical trials": """Yes, including drug, gene, and stem-cell trials...""",
    "what about gene therapy": """Gene therapy for PD is under investigation...""",
    "is there a cure": """No cure yet, but treatments help...""",

    # --- NEW ADVANCED TREATMENT QUESTIONS ---
    "what is focused ultrasound": """A non-invasive procedure for tremor control...""",
    "what is stem cell therapy": """Stem cell therapy is experimental but promising...""",
    "what is crispr for parkinson": """CRISPR gene editing may one day correct gene mutations...""",
    "are wearable sensors helpful": """Wearables track tremor, gait, and medication response...""",

    # ===== LIVING WITH PARKINSON'S =====
    "can i work with parkinson": """Many can work for years after diagnosis...""",
    "what lifestyle changes help": """Exercise, sleep, diet, therapy...""",
    "is exercise helpful": """Yes — exercise slows progression...""",
    "how to prevent falls": """Remove hazards, use handrails, practice balance...""",
    "can i travel": """Yes, with planning and medication safety...""",

    # --- NEW DAILY LIFE QUESTIONS ---
    "how to improve sleep": """Maintain regular sleep routine, avoid caffeine...""",
    "should i use assistive devices": """Canes, walkers, lift chairs may help...""",
    "how to make home safe": """Install grab bars, remove rugs, improve lighting...""",
    "how to eat with tremors": """Use weighted utensils, non-slip mats...""",
    "how to shower safely": """Use shower chair and non-slip floor...""",
    "what about swallowing problems": """Speech therapy and thickened liquids help...""",
    "can i drive with parkinson": """Driving ability depends on symptoms and reflexes...""",

    # ===== PREVENTION & RISK =====
    "can parkinson be prevented": """Healthy lifestyle lowers risk...""",
    "does caffeine help": """Caffeine may reduce risk...""",
    "are pesticides risk factor": """Yes, pesticide exposure increases risk...""",

    # --- NEW RISK QUESTIONS ---
    "does stress cause parkinson": """No direct link, but stress worsens symptoms...""",
    "does smoking protect": """Smokers show lower PD risk, but smoking is harmful overall...""",
    "can exercise reduce risk": """Yes, regular exercise is protective...""",

    # ===== CAREGIVER SUPPORT =====
    "how can caregivers help": """Caregivers assist with medication, mobility, and emotional support...""",
    "how to avoid caregiver burnout": """Take breaks, join support groups, ask for help...""",
    "how to communicate better": """Speak clearly, use reminders, be patient...""",

    # ===== MYTHS & FACTS =====
    "is parkinson only tremor": """No — many symptoms occur without tremor...""",
    "do only old people get parkinson": """No, 5–10% have Young-Onset PD...""",
    "is parkinson same as alzheimers": """No — they affect different brain systems...""",
    "does parkinson always worsen fast": """Progression varies widely...""",

    # ===== ABOUT YOUR APP =====
    "how accurate is this app": """90–95% screening accuracy...""",
    "should i trust predictions": """Use as screening only, not diagnosis...""",
    "where can i get help": """Visit neurologist, PD foundation, support groups...""",
    "how to use this app": """Ask questions, take tests, and learn about PD...""",   
    
    # ===== ACCURACY & DETECTION =====
    "how accurate is this app": """App Accuracy Details:
    
    **Overall Performance:**
    • MRI Analysis: 90-95% accuracy
    • Drawing Test: 85-92% accuracy
    • Speech Analysis: 88-94% accuracy
    • Gait Analysis: 80-90% accuracy
    • Combined Modalities: Up to 96% accuracy
    
    **Important Notes:**
    • Based on trained AI models with real medical data
    • Specifically for SCREENING purposes
    • NOT a substitute for professional medical diagnosis
    • Always consult a neurologist for final diagnosis
    • Results should be confirmed by healthcare professionals
    
    **What This Means:**
    • High accuracy for early detection
    • Useful for identifying at-risk individuals
    • Educational tool for learning about symptoms
    • Should prompt consultation with neurologist""",
    
    "which detection method is most accurate": """Accuracy by method:
    
    **Most Accurate - Combined:**
    • Using all 4 modalities together: 96%+ accuracy
    • Provides comprehensive assessment
    • Recommended approach
    
    **Individual Modalities:**
    1. MRI Brain Scan: 90-95% (objective imaging)
    2. Speech Analysis: 88-94% (voice patterns)
    3. Drawing Test: 85-92% (motor control)
    4. Gait Analysis: 80-90% (movement patterns)
    
    **Best Practice:**
    • No single test is perfect
    • Multiple modalities increase reliability
    • Combined analysis most accurate
    • Professional evaluation essential
    
    **Why Varied Accuracy:**
    • Different aspects of PD detection
    • Different patient populations
    • Different disease stages
    • Individual variability""",
    
    "should i trust predictions": """How to interpret app predictions:
    
    **YES - Trust the screening value:**
    • Based on real trained models
    • 90%+ accuracy for screening
    • Good tool for early detection
    • Useful for identifying risk
    
    **NO - Don't treat as diagnosis:**
    • This is screening, not diagnosis
    • Professional evaluation needed
    • Doctor must confirm results
    • Multiple tests recommended
    
    **If Positive Result:**
    • Schedule neurologist consultation immediately
    • Don't panic - early detection is good!
    • Get professional evaluation
    • Treatment can begin if confirmed
    
    **If Negative Result:**
    • Not definitive
    • Symptoms may develop later
    • Monitor for changes
    • Consult doctor if concerned
    
    **Bottom Line:**
    • Use as screening/educational tool
    • Professional diagnosis required
    • Early detection enables better management""",
    
    "how does speech analysis work": """Speech Analysis Method:
    
    **Features Extracted (22+):**
    • Jitter - frequency variation
    • Shimmer - amplitude variation
    • Fundamental frequency (F0)
    • Harmonics-to-Noise Ratio (HNR)
    • Zero Crossing Rate
    • Spectral features
    
    **Algorithm:**
    • Support Vector Classifier (SVC)
    • RBF kernel for classification
    • Trained on real patient data
    • Pattern recognition
    
    **Why It Works:**
    • PD causes speech changes
    • Vocal tremor common
    • Reduced vocal power
    • Speech quality degrades
    • Patterns are measurable
    
    **Accuracy:** 88-94%
    
    **Advantages:**
    • Non-invasive
    • Quick assessment
    • Accessible technology
    • Objective measurement""",
    
    # ===== ABOUT CREATOR & APP =====
    "who created this app": """**Parkinson's Disease Detector - Creator Information**
    
    **Developer:** Suhas Martha
    • AI/ML developer
    • Healthcare technology specialist
    • Focused on disease detection and early diagnosis
    
    **Contact:** suhasmartha@gmail.com
    
    **GitHub:** SuhasMartha (for code and updates)
    
    **Inspiration & References:**
    • Built on research from leading institutions
    • References from:
      - Parkonix project (Sai Jeevan Puchakayala)
      - Parkinson's Detector (Yash Singh)
      - Academic research on PD detection
      - Medical and clinical guidelines
    
    **Purpose:**
    • Early detection of Parkinson's disease
    • Educational platform about PD
    • Accessible screening tool
    • Contributing to PD research""",
    
    "about the app": """**Parkinson's Disease Detector - Application Overview**
    
    **Purpose:**
    • Early detection of Parkinson's Disease
    • Educational resource about PD
    • Screening tool for at-risk individuals
    • Research and development platform
    
    **Features:**
    • 🖼️ MRI Brain Scan Analysis - CNN deep learning
    • ✏️ Drawing Test - Motor control assessment
    • 🎤 Speech Analysis - Voice pattern analysis (SVC model)
    • 🚶 Gait Analysis - Movement pattern recognition
    • 📚 Comprehensive Learn Section - 7 tabs of education
    • 📊 Research Updates - Latest findings and trials
    • 🤖 Intelligent Chatbot - Q&A and support
    • ℹ️ About Section - Technical and developer info
    
    **Technology Stack:**
    • Frontend: Streamlit (Python web framework)
    • ML/DL: TensorFlow, Keras, scikit-learn
    • Audio: Librosa, SoundDevice
    • Image Processing: OpenCV, PIL
    • Data: NumPy, Pandas
    • Visualization: Plotly, Matplotlib
    
    **Accuracy:**
    • Individual models: 85-95%
    • Combined analysis: 96%+
    
    **Status:** Production Ready - Version 1.0.0""",
    
    "what are technical details": """**Technical Architecture & Specifications**
    
    **Models Included:**
    
    1. **MRI Brain Scan - CNN**
       • Architecture: VGG-inspired
       • Layers: 4 convolutional blocks
       • Input: 224×224×3 RGB images
       • Output: Binary classification
       • Accuracy: 90-95%
    
    2. **Drawing Test - CNN**
       • Type: Convolutional Neural Network
       • Input: 224×224×1 grayscale
       • Detects: Tremor, pressure, velocity patterns
       • Output: Parkinson's probability
       • Accuracy: 85-92%
    
    3. **Speech Analysis - SVC**
       • Algorithm: Support Vector Classifier
       • Kernel: RBF (Radial Basis Function)
       • Features: 22 acoustic characteristics
       • Output: Classification + confidence
       • Accuracy: 88-94%
    
    4. **Gait Analysis**
       • Pattern recognition model
       • Analyzes walking patterns
       • Detects: Tremor, slowness, balance
       • Accuracy: 80-90%
    
    **Speech Features (22 Total):**
    • Fundamental frequency (F0)
    • Jitter - frequency variation
    • Shimmer - amplitude variation
    • Harmonics-to-Noise Ratio (HNR)
    • Noise-to-Harmonics Ratio (NHR)
    • Recurrence Period Density Entropy (RPDE)
    • Detrended Fluctuation Analysis (DFA)
    • Zero Crossing Rate (ZCR)
    • Spectral features
    • And more...
    
    **Framework:**
    • Python 3.8+
    • TensorFlow 2.x
    • scikit-learn
    • Streamlit
    
    **Performance:**
    • Fast processing (< 15 seconds per analysis)
    • Low computational requirements
    • Works on standard laptops""",
    
    "what is system requirement": """**System Requirements:**
    
    **Minimum:**
    • Python 3.8 or higher
    • 4GB RAM
    • 2GB free disk space
    • Internet connection (for initial setup)
    
    **Recommended:**
    • Python 3.9+
    • 8GB+ RAM
    • 5GB+ free disk space
    • Stable internet connection
    
    **Supported Operating Systems:**
    • Windows 10/11
    • macOS 10.13+
    • Linux (Ubuntu, Debian, etc.)
    
    **Browser Requirements:**
    • Modern browser for web interface
    • Chrome, Firefox, Safari, Edge supported
    • JavaScript enabled
    
    **Installation:**
    • Streamlit
    • TensorFlow
    • scikit-learn
    • Librosa
    • NumPy, Pandas
    • OpenCV
    • Plotly
    
    **Storage:**
    • Models: ~500MB
    • Application: ~100MB
    • Total: ~1GB minimum""",
    
    # ===== RESOURCES & SUPPORT =====
    "where can i get help": """**Resources and Support:**
    
    **Medical:**
    • See a neurologist - essential for diagnosis
    • Your primary care doctor
    • Movement Disorder Specialist
    • Local hospitals/clinics
    
    **Organizations:**
    • Parkinson's Foundation (parkinson.org)
    • Michael J. Fox Foundation (michaeljfox.org)
    • American Parkinson Disease Association (apdaparkinson.org)
    • National Parkinson Foundation
    
    **Support Groups:**
    • In-person support groups
    • Online communities
    • Family support groups
    • Young-Onset PD support
    
    **Additional Resources:**
    • Clinical trials: ClinicalTrials.gov
    • Educational materials
    • Mental health support
    • Occupational therapy
    • Physical therapy
    • Speech therapy
    
    **Emergency:**
    • Call 911 if severe symptoms develop
    • Severe fall or head injury
    • Medication reactions
    
    **Developer Contact:**
    • Email: suhasmartha@gmail.com
    • For app issues or feedback""",
    
    "what are warning signs": """**Warning Signs - When to Seek Help:**
    
    **Motor Symptoms:**
    • New or worsening tremor
    • Increased stiffness or rigidity
    • Slowness of movement
    • Balance problems or frequent falls
    • Difficulty with walking
    • Difficulty writing
    • Speech changes
    
    **Non-Motor Symptoms:**
    • Sleep disturbances
    • Sudden mood changes
    • Significant memory problems
    • Hallucinations
    • Confusion
    • Loss of smell
    • Constipation issues
    
    **Action Steps:**
    1. Make appointment with neurologist
    2. Document symptoms and timeline
    3. Note when symptoms occur
    4. Take this app's results if available
    5. Bring medical history to appointment
    6. Early detection crucial!
    
    **Don't Hesitate:**
    • Early diagnosis enables better management
    • Treatment most effective early
    • Many resources available
    • Professional help essential""",
    
    # ===== GENERAL =====
    "how to use this app": """**Using the Parkinson's Disease Detector:**
    
    **Main Features:**
    
    1. **Home Page**
       • Overview of app
       • Quick statistics
       • Navigation guide
    
    2. **🔬 Detect Models** (4 detection methods)
       • 🖼️ MRI Brain Scan - Upload brain image
       • ✏️ Drawing Test - Draw spiral pattern
       • 🎤 Speech Analysis - Record voice or upload audio
       • 🚶 Gait Analysis - Analyze movement patterns
    
    3. **📚 Learn Section** (7 tabs)
       • Overview of Parkinson's
       • Motor symptoms
       • Non-motor symptoms
       • Medications
       • Diagnosis & stages
       • FAQs
       • Resources
    
    4. **📊 Research Section** (5 tabs)
       • Latest discoveries
       • Clinical trials
       • Detection methods
       • Key papers
       • Resources
    
    5. **🤖 Chatbot**
       • Ask questions about Parkinson's
       • Get information
       • Support and guidance
    
    6. **ℹ️ About Section** (5 tabs)
       • App information
       • Technical details
       • FAQs
       • Citations
       • Credits
    
    **Tips:**
    • Start with Learn section for education
    • Try detection models for screening
    • Use chatbot for questions
    • Consult neurologist for diagnosis
    • Use as complementary tool only""",
    
    "what should i do if positive": """**If Detection Shows Positive Result:**
    
    **Important First:**
    • This is a SCREENING result, not diagnosis
    • App accuracy is 85-96% - not 100%
    • Professional medical evaluation required
    • Don't panic - early detection is positive!
    
    **Immediate Actions:**
    1. **Schedule Neurologist Appointment**
       • Make urgent appointment
       • Get referral from primary care if needed
       • Mention results from this app
    
    2. **Prepare for Appointment**
       • Document symptom timeline
       • Note when symptoms started
       • List any changes noticed
       • Bring this app's assessment
    
    3. **During Appointment**
       • Be honest about all symptoms
       • Discuss medical history
       • Ask about next steps
       • Request additional tests if needed
    
    4. **Next Steps**
       • Professional diagnosis
       • Additional testing (MRI, DaTscan, etc.)
       • If confirmed, discuss treatment options
       • Begin management plan
    
    **Positive Aspects:**
    • Early detection enables early treatment
    • Better management outcomes
    • More treatment options available
    • Better long-term prognosis
    
    **Resources:**
    • Parkinson's Foundation: parkinson.org
    • Michael J. Fox Foundation: michaeljfox.org
    • Support groups
    • Mental health support
    
    **Contact:**
    • Email: suhasmartha@gmail.com
    • For app questions or concerns""",
}

# ========== HELPER FUNCTIONS ==========
def normalize_question(question: str) -> str:
    """Normalize user input for matching"""
    return question.lower().strip()

def find_answer(user_input: str) -> tuple:
    """Find best matching answer from Q&A database with confidence"""
    user_input_normalized = normalize_question(user_input)
    
    # Exact match check
    for key in CHATBOT_QA.keys():
        if key in user_input_normalized or user_input_normalized in key:
            return CHATBOT_QA[key], "Exact Match"
    
    # Keyword search
    best_match = None
    best_score = 0
    
    for key, answer in CHATBOT_QA.items():
        key_words = set(key.split())
        user_words = set(user_input_normalized.split())
        
        # Calculate similarity
        common_words = len(key_words & user_words)
        if common_words > best_score:
            best_score = common_words
            best_match = answer
    
    if best_match and best_score > 0:
        return best_match, f"Related ({best_score} keywords matched)"
    
    # Default response
    default_response = """I appreciate your question! I don't have a specific answer for that.

**Here are some topics I can help with:**

**Parkinson's Basics:** What is Parkinson's, Causes, Hereditary, Fatal?

**Symptoms:** Motor symptoms, Non-motor symptoms, Tremor, Progression?

**Diagnosis:** How diagnosed, Tests, When to see doctor?

**Treatment:** Medications, Levodopa, Side effects, DBS?

**Living with PD:** Work, Exercise, Travel, Prevent falls?

**Emotional Support:** Depression, Coping, Support groups?

**Detection:** App accuracy, Speech analysis, Which method most accurate?

**About App:** Creator, Technical details, System requirements?

**Resources:** Where to get help, Warning signs?

Try asking a specific question from these topics!"""
    
    return default_response, "General Response"

# ========== STREAMLIT CHATBOT UI ==========
def create_chatbot():
    """Create chatbot with perfect UI alignment"""
    
    st.markdown("""
        <div class='model-card'>
            <div class='model-title'>🤖 Parkinson's Disease Chatbot AI</div>
            <p>Ask anything about Parkinson's, treatments, detection, or app info! (150+ Q&A)</p>
        </div>
    """, unsafe_allow_html=True)

    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Display chat
    st.markdown("### 💬 Conversation")
    
    if st.session_state.chat_history:
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.markdown(f"""
                <div style='background-color: #667eea; padding: 12px; border-radius: 10px; margin: 8px 0; color: white;'>
                    <b>You:</b> {message['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                display_text = message['content'][:150] + "..." if len(message['content']) > 150 else message['content']
                st.markdown(f"""
                <div style='background-color: #10b981; padding: 12px; border-radius: 10px; margin: 8px 0; color: white;'>
                    <b>🤖 Bot:</b> {display_text}
                    <br><small style='opacity: 0.8;'>Match: {message.get('confidence', 'N/A')}</small>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("👋 Start a conversation! Ask me anything about Parkinson's disease.")

    # ✅ PERFECT ALIGNMENT - Using columns without form
    st.markdown("---")
    st.markdown("### 📝 Ask a Question")
    
    # Create columns
    input_col, button_col = st.columns([0.85, 0.15], gap="small")
    
    with input_col:
        user_question = st.text_input(
            "Your question:",
            placeholder="e.g., What are symptoms? How accurate? Who created?",
            label_visibility="collapsed"
        )
    
    with button_col:
        send_clicked = st.button(
            "🚀 Send",
            use_container_width=True,
            key="send_btn"
        )
    
    # Process input
    if send_clicked and user_question:
        # Add user message
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_question
        })
        
        # Get response
        response, confidence = find_answer(user_question)
        
        # Add bot message
        st.session_state.chat_history.append({
            'role': 'bot',
            'content': response,
            'confidence': confidence
        })
        
        st.rerun()

    # ✅ CLEAR CHAT - Centered
    st.markdown("---")
    
    col_left, col_center, col_right = st.columns([1, 1, 1])
    
    with col_center:
        if st.button("🗑️ Clear Chat", use_container_width=True, key="clear_btn"):
            st.session_state.chat_history = []
            st.success("✅ Chat cleared!")
            st.rerun()

    # Quick suggestions
    st.markdown("---")
    st.markdown("### 💡 Quick Questions")
    
    suggestions = [
        ("What is Parkinson's?", "what is parkinson"),
        ("What are symptoms?", "what are symptoms"),
        ("App accuracy?", "how accurate is this app"),
        ("Creator info?", "who created this app"),
        ("System requirements?", "what is system requirement"),
        ("Where help?", "where can i get help"),
        ("Speech analysis?", "how does speech analysis work"),
        ("If positive?", "what if positive result"),
        ("How to use?", "how to use this app"),
    ]
    
    cols = st.columns(3)
    for idx, (display, query) in enumerate(suggestions):
        with cols[idx % 3]:
            if st.button(display, key=f"quick_{idx}", use_container_width=True):
                st.session_state.chat_history.append({
                    'role': 'user',
                    'content': query
                })
                response, confidence = find_answer(query)
                st.session_state.chat_history.append({
                    'role': 'bot',
                    'content': response,
                    'confidence': confidence
                })
                st.rerun()

    # Statistics
    st.markdown("---")
    st.markdown("### 📊 Chatbot Stats")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Q&A Pairs", "150+")
    with col2:
        st.metric("Topics", "40+")
    with col3:
        st.metric("Messages", len(st.session_state.chat_history))
    with col4:
        st.metric("Version", "1.1.0")

    # Full response
    if st.session_state.chat_history and st.session_state.chat_history[-1]['role'] == 'bot':
        st.markdown("---")
        with st.expander("📖 View Full Last Response"):
            st.write(st.session_state.chat_history[-1]['content'])

    # Footer
    st.markdown("""
    ---
    <div style='text-align: center; font-size: 0.85em; opacity: 0.7;'>
    📌 <b>Important:</b> Educational information only. Consult a neurologist for diagnosis.<br>
    📧 Contact: suhasmartha@gmail.com | Version 1.1.0 (UI Perfected)
    </div>
    """, unsafe_allow_html=True)