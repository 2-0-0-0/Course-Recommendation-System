🎓 Personalized Course Recommendation System
A matrix factorization–based recommendation engine that suggests tailored online courses to users by analyzing historical enrollment patterns and inferring user preferences from latent behavioral features.


📘 Project Overview
This project builds a content-agnostic recommender system that intelligently ranks and recommends courses to users based on the latent similarities between user enrollment behavior and course characteristics, even in the absence of explicit ratings.

Key goals:

Handle cold-start scenarios via enrollment similarity.

Capture hidden structures in user-course interaction using matrix factorization.

Make scalable, interpretable recommendations for new users.

📊 Dataset
The dataset contains:

title: Course name

enrollment_number: Simulated user ID or behavior signature

rating: Optional—average course ratings for score refinement

Optional metadata like subject, platform, etc.

Basic Statistics:
~6,599 unique course titles

4,858 simulated users (enrollments)

10 latent dimensions for course and user vectors

🔍 EDA and Insights
User Behavior Distribution

Enrollment numbers follow a skewed distribution.

A small set of users enrolled in many courses.

Course Popularity

The majority of courses have fewer than 100 enrollments.

"Long-tail" behavior observed in course engagement.

Missing Values

Cleaned/filtered based on presence of title or rating.

Visuals

Histograms, box plots, and heatmaps for quality assessment

Optional PCA or TSNE on course vectors to visualize clustering

🧠 Modeling Approach
1️⃣ Matrix Factorization
We decompose the user_course_matrix into:

enrollment_factors: user embeddings

course_profiles: course embeddings

Each entity is mapped to a 10-dimensional latent space.

2️⃣ Cold Start Matching
New users are mapped to the nearest existing enrollment profile using:

python
closest_enrollment = np.abs(existing_enrollments - new_enrollment).argmin()
3️⃣ Scoring Function
Prediction via dot product between user and course vectors:

python
predicted_scores = user_vector.dot(course_profiles.T)
Scores are ranked and optionally normalized using MinMaxScaler.

4️⃣ Score Refinement (Optional)
Multiply predicted scores with normalized course ratings:

python
final_scores = predicted_scores * course_rating_normalized
⚙️ Key Functionality
python
def recommend_courses_for_enrollment(enrollment_number, top_n=5):
    ...
    return top_n_ranked_courses
Accepts a user's approximate enrollment behavior

Returns top-N predicted course titles personalized to that user


Clean and normalize columns: title, enrollment_number, etc.

Run the notebook or Python script:

python
recommendations = recommend_courses_for_enrollment(50000, top_n=5)
📈 Results and Evaluation
Top 5 personalized courses returned with confidence scores.

Scores can be normalized or enriched with content/rating similarity.

Example output:

1. Advanced Manufacturing Enterprise             0.98  
2. Data Science for Professionals                0.87  
3. Pattern Discovery in Data Mining              0.84  
