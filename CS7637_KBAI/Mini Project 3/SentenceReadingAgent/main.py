# from SentenceReadingAgent import SentenceReadingAgent
#
# def test():
#     #This will test your SentenceReadingAgent
# 	#with nine initial test cases.
#
#     test_agent = SentenceReadingAgent()
#
#     sentence_1 = "Ada brought a short note to Irene."
#     question_1 = "Who brought the note?"
#     question_2 = "What did Ada bring?"
#     question_3 = "Who did Ada bring the note to?"
#     question_4 = "How long was the note?"
#
#     sentence_2 = "David and Lucy walk one mile to go to school every day at 8:00AM when there is no snow."
#     question_5 = "Who does Lucy go to school with?"
#     question_6 = "Where do David and Lucy go?"
#     question_7 = "How far do David and Lucy walk?"
#     question_8 = "How do David and Lucy get to school?"
#     question_9 = "At what time do David and Lucy walk to school?"
#
#     sentence_3 = "Serena ran a mile this morning."
#     question_10 = "What did Serena run?"
#
#     print(test_agent.solve(sentence_1, question_1))  # "Ada"
#     print(test_agent.solve(sentence_1, question_2))  # "note" or "a note"
#     print(test_agent.solve(sentence_1, question_3))  # "Irene"
#     print(test_agent.solve(sentence_1, question_4))  # "short"
#
#     print(test_agent.solve(sentence_2, question_5))  # "David"
#     print(test_agent.solve(sentence_2, question_6))  # "school"
#     print(test_agent.solve(sentence_2, question_7))  # "mile" or "a mile"
#     print(test_agent.solve(sentence_2, question_8))  # "walk"
#     print(test_agent.solve(sentence_2, question_9))  # "8:00AM"
#     print(test_agent.solve(sentence_3, question_10))
#
# if __name__ == "__main__":
#     test()

import time
import matplotlib.pyplot as plt
import numpy as np
from SentenceReadingAgent import SentenceReadingAgent


def test():
    test_agent = SentenceReadingAgent()

    test_cases = [
        # Your test cases
        ("Ada brought a short note to Irene.", "Who brought the note?"),
        ("Ada brought a short note to Irene.", "What did Ada bring?"),
        ("Ada brought a short note to Irene.", "Who did Ada bring the note to?"),
        ("Ada brought a short note to Irene.", "How long was the note?"),
        ("David and Lucy walk one mile to go to school every day at 8:00AM when there is no snow.",
         "Who does Lucy go to school with?"),
        ("David and Lucy walk one mile to go to school every day at 8:00AM when there is no snow.",
         "Where do David and Lucy go?"),
        ("David and Lucy walk one mile to go to school every day at 8:00AM when there is no snow.",
         "How far do David and Lucy walk?"),
        ("David and Lucy walk one mile to go to school every day at 8:00AM when there is no snow.",
         "How do David and Lucy get to school?"),
        ("David and Lucy walk one mile to go to school every day at 8:00AM when there is no snow.",
         "At what time do David and Lucy walk to school?"),
        ("Serena ran a mile this morning.", "What did Serena run?"),
        ("Serena saw a home last night with her friend.", "Who was with Serena?"),
        ("There are a thousand children in this town.", "Who is in this town?"),
        ("The island is east of the city.", "What is east of the city?"),
        ("Give us all your money.", "Who should you give your money to?"),
        ("The blue bird will sing in the morning.", "When will the bird sing?"),
        ("The blue bird will sing in the morning.", "What will sing in the morning?"),
        ("Bring the letter to the other room.", "What should be brought to the other room?"),
        ("The island is east of the city.", "Where is the island?"),
        ("The red fish is in the river.", "What is in the river?"),
        ("The house is made of paper.", "What is made of paper?"),
        ("This year David will watch a play.", "Who will watch a play?")
    ]

    # Track time taken for each question
    times = []
    question_numbers = list(range(1, len(test_cases) + 1))  # Assign numbers to questions for simplicity

    for i, (sentence, question) in enumerate(test_cases):
        start_time = time.time()
        answer = test_agent.solve(sentence, question)
        end_time = time.time()

        # Record the time taken
        times.append(end_time - start_time)

        print(f"Question {i + 1}: {question}\nAnswer: {answer}\nTime taken: {end_time - start_time:.6f} seconds\n")

    # Normalize the times to make smaller differences more visible
    times_standardized = np.array(times) / np.max(times)

    # Plot the results with the correct labeling
    plt.figure(figsize=(10, 6))
    plt.barh([f"Q{i}" for i in question_numbers], times_standardized, color="skyblue")
    plt.xlabel('Standardized Time Taken (Normalized)')
    plt.ylabel('Questions (by difficulty)')
    plt.title('Standardized Time Taken to Solve Each Question')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test()
