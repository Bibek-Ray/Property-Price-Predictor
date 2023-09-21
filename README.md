# Property-Price-Predictor

This is a Flask web application that predicts the price of a house based on various features. It utilizes a trained machine learning model to make predictions.

Features includes:

- Preferred Area
- Availability of Garage
- Furnishing
- Number of stories
- Basement
- Mainroad
- Guestroom
- Hot water heating
- Air Conditioning
- Area
- Number of Bedrooms
- Number of Bathrooms

The Model is trained using Lasso Regression model. Lasso regression is a linear regression technique that incorporates regularization to improve the model's performance and prevent overfitting. It adds a penalty term to the cost function, which helps in shrinking the coefficients of less important features towards zero, effectively performing feature selection. It has an Adjusted R square value of 0.70267 which is considered good.

## Demo

https://github.com/Bibek-Ray/Property-Price-Predictor/assets/82962780/27e00cc4-d962-4c0f-b31b-6d75e8782ae1

## Getting Started

To get started with the House Price Predictor, follow these steps:

1. Clone the repository:
   git clone https://github.com/Bibek-Ray/Property-Price-Predictor.git

2. Install the required dependencies:
   pip install -r requirements.txt

3. Run the Flask application:
   python app.py

4. Open your web browser and go to `http://localhost:5000` to access the application.

## Usage

1. Enter the required details in the input fields on the web page, such as the area, number of bedrooms, number of bathrooms, and location.
2. Click on the "Submit" button to generate the house price prediction.
3. The predicted price will be displayed on the web page.

## Technologies Used

- Python
- Flask
- HTML/CSS/Javascript
- Machine Learning (scikit-learn)

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.


   




