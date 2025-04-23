# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

from flask import Flask
import recommendations
app = Flask(__name__)

@app.route('/<int:userId>/recommendations')
def show_recommendations(userId):
    return [
        {"name" : "The Matrix", "id" : 1, "genres" : ["action", "sci-fi"], "rating" : 4.28, "ratings" : 19008},
        {"name" : "Toy Story", "id" : 2, "genres" : ["animation"], "rating" : 3.87, "ratings" : 12003},
        {"name" : "The Godfather II", "id" : 3, "genres" :["epic", "drama"], "rating" : 4.73, "ratings" : 5007}
    ]

@app.route('/recommendations/<genre>')
def show_recommendations_genre(genre):
    return recommendations.build_chart(genre, 0.85).to_json(orient='records')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
@app.route('/recommendations/title/<title>')
def show_recommendations_title(title):
    return recommendations.get_recommendations(title).to_json(orient='records')

@app.route('/recommendations/title/improved/<title>')
def show_recommendations_improved_title(title):
    return recommendations.improved_recommendations(title).to_json(orient='records')

@app.route('/recommendations/<userId>/<title>')
def show_recommendations_hybrid(userId, title):
    return recommendations.hybrid(userId, title).to_json(orient='records')