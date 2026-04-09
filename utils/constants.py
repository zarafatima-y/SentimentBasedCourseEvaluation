# Aspect keywords (from your code)
ASPECT_KEYWORDS = {
    "instructor": [
        "professor", "teacher", "lecturer", "ta", "tutor", "instructor",
        "they explained", "he taught", "she lectured", "office hours",
        "answered questions", "responsive", "approachable", "knowledgeable",
        "passionate", "enthusiastic", "caring", "helpful person", "great teacher"
    ],
    "course_content": [
        "topics", "subject", "curriculum", "syllabus", "what we covered",
        "concepts", "material", "content was", "things we learned",
        "scope", "coverage", "depth", "breadth", "topics included"
    ],
    "assignments_labs": [
        "assignments", "homework", "labs", "projects", "exercises",
        "problem sets", "weekly work", "tasks", "deliverables",
        "instructions were", "guidelines", "what we had to do",
        "the work", "these assignments", "each lab"
    ],
    "assessments": [
        "exams", "tests", "midterm", "final", "quizzes",
        "grading", "grades", "marks", "scoring", "rubric",
        "how we were graded", "the exam", "test questions",
        "curve", "grade distribution"
    ],
    "workload_pace": [
        "workload", "pace", "speed", "time commitment",
        "hours per week", "effort", "how fast", "too slow",
        "rushed", "crammed", "manageable", "heavy load",
        "light workload", "time consuming", "took forever"
    ],
    "learning_outcomes": [
        "learned", "understand now", "grasped", "figured out",
        "skills gained", "know how to", "can apply", "feel confident",
        "improved at", "growth", "development", "got better at",
        "ready for", "prepared me", "helped me learn"
    ],
    "engagement_interest": [
        "engaging", "interesting", "boring", "exciting", "fun",
        "dull", "captivating", "held my attention", "kept me interested",
        "made me want to", "participate", "interactive", "stimulating"
    ],
    "difficulty_challenge": [
        "difficult", "hard", "easy", "challenging", "tough",
        "simple", "demanding", "rigorous", "struggled", "struggle",
        "too hard", "too easy", "just right", "difficulty level"
    ],
    "resources_materials": [
        "textbook", "book", "readings", "slides", "notes",
        "handouts", "videos", "recordings", "website", "online platform",
        "eclass", "zybooks", "materials", "resources", "supplements"
    ],
    "support_help": [
        "help", "assistance", "support", "guidance", "tutoring",
        "extra help", "office hours helped", "they helped",
        "answered my questions", "quick response", "slow response",
        "available when", "accessible", "mentor", "guided me"
    ],
    "practical_application": [
        "practical", "hands-on", "real world", "real life",
        "apply to", "use in", "relevant to job", "useful for work",
        "applicable", "practically", "implement", "execute"
    ],
    "overall_experience": [
        "overall", "generally", "in general", "would recommend",
        "worth it", "worthwhile", "good course", "great class",
        "bad experience", "loved it", "hated it", "enjoyed",
        "positive experience", "negative experience"
    ]
}

# Emotion to sentiment mapping
EMOTION_TO_SENTIMENT = {
    'neutral': 'neutral',
    'joy': 'positive',
    'surprise': 'positive',
    'sadness': 'negative',
    'anger': 'negative',
    'fear': 'negative',
    'disgust': 'negative'
}

# Negative emotions for analysis
NEGATIVE_EMOTIONS = ['anger', 'sadness', 'fear', 'disgust']