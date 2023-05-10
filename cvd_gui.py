import tkinter as tk
from cvd_model import *

class CVD_GUI:
    def __init__(self):

        # Create main window
        self.main_window = tk.Tk()
        self.main_window.title("Student Performance Prediction")

        # Create multiple frames to group widgets 
        self.one_frame = tk.Frame()
        self.two_frame = tk.Frame()
        self.three_frame = tk.Frame()
        self.four_frame = tk.Frame()
        self.five_frame = tk.Frame()
        self.six_frame = tk.Frame()
        self.seven_frame = tk.Frame()
        self.eight_frame = tk.Frame()
        self.nine_frame = tk.Frame()
        self.ten_frame = tk.Frame()

        # Create widget for one frame - display title
        self.title_label = tk.Label(self.one_frame, text = "STUDENT PERFORMANCE PREDICITION")
        self.title_label.pack()

        # Create widget for two frame - gender input
        self.gender_label = tk.Label(self.two_frame, text = "Gender: ")
        self.gender_label.pack(side = 'left')
        self.click_gender_var = tk.StringVar()
        self.click_gender_var.set ("Male")
        self.gender_inp = tk.OptionMenu(self.two_frame, self.click_gender_var, "Male", "Female")
        self.gender_inp.pack(side = 'left')

        # Create widget for three frame - race/ethnicity group input
        self.race_label = tk.Label(self.three_frame, text = "Race/Ethnicity group: ")
        self.race_label.pack(side = 'left')
        self.click_race_var = tk.StringVar()
        self.click_race_var.set ("Group A")
        self.race_inp = tk.OptionMenu(self.three_frame, self.click_race_var, "Group A", "Group B", "Group C", "Group D", "Group E")
        self.race_inp.pack(side = 'left')

        # Create widget for four frame - parent education input
        self.parent_label = tk.Label(self.four_frame, text = "Parent Education: ")
        self.parent_label.pack(side = 'left')
        self.click_parent_var = tk.StringVar()
        self.click_parent_var.set ("Associate's Degree")
        self.parent_inp = tk.OptionMenu(self.four_frame, self.click_parent_var, "Associate's Degree", "Bachelor's Degree", 
                                        "High School", "Master's Degree", "Some College", "Some High School")
        self.parent_inp.pack(side = 'left')

        # Create widget for five frame - lunch input
        self.lunch_label = tk.Label(self.five_frame, text = "Lunch: ")
        self.lunch_label.pack(side = 'left')
        self.click_lunch_var = tk.StringVar()
        self.click_lunch_var.set ("Standard")
        self.lunch_inp = tk.OptionMenu(self.five_frame, self.click_lunch_var, "Standard", "Free/Reduced")
        self.lunch_inp.pack(side = 'left')

        # Create widget for six frame - test prep course input
        self.course_label = tk.Label(self.six_frame, text = "Test Prep Course: ")
        self.course_label.pack(side = 'left')
        self.click_course_var = tk.StringVar()
        self.click_course_var.set ("None")
        self.course_inp = tk.OptionMenu(self.six_frame, self.click_course_var, "Completed", "None")
        self.course_inp.pack(side = 'left')

        # Create the widgets for seven frame - math score input
        self.math_label = tk.Label(self.seven_frame, text = 'Math Score:')
        self.math_entry = tk.Entry(self.seven_frame, bg = "white", fg = "black", width = 10)
        self.math_entry.insert(0,'0')
        self.math_label.pack(side='left')
        self.math_entry.pack(side='left')

        # Create the widgets for eight frame - reading score input
        self.read_label = tk.Label(self.eight_frame, text = 'Reading Score:')
        self.read_entry = tk.Entry(self.eight_frame, bg = "white", fg = "black", width = 10)
        self.read_entry.insert(0,'0')
        self.read_label.pack(side='left')
        self.read_entry.pack(side='left')

        # Create the widgets for nine frame - writing score input
        self.write_label = tk.Label(self.nine_frame, text = 'Writing Score:')
        self.write_entry = tk.Entry(self.nine_frame, bg = "white", fg = "black", width = 10)
        self.write_entry.insert(0,'0')
        self.write_label.pack(side='left')
        self.write_entry.pack(side='left')

        # Create the widgets for ten frame - test (preduction of pass or no pass)
        self.test_predict_ta = tk.Text(self.ten_frame, height = 10, width = 25, bg = 'light green')
        self.test_predict_ta.pack(side = 'left')

        # Create predict button
        self.btn_predict = tk.Button(self.ten_frame, text = 'Predict Student Performance', command = self.predict_test)
        self.btn_predict.pack()

        # Create quit button
        self.btn_quit = tk.Button(self.ten_frame, text = 'Quit', command = self.main_window.destroy)
        self.btn_quit.pack()

        # Pack all the frames 
        self.one_frame.pack()
        self.two_frame.pack()
        self.three_frame.pack()
        self.four_frame.pack()
        self.five_frame.pack()
        self.six_frame.pack()
        self.seven_frame.pack()
        self.eight_frame.pack()
        self.nine_frame.pack()
        self.ten_frame.pack()

        # Enter the tkinter main loop
        tk.mainloop()

    def predict_test(self):
        result_string = ''
        self.test_predict_ta.delete(0.0, tk.END)

        student_gender = self.click_gender_var.get()
        if (student_gender == "Male"):
            student_gender = 1
        else:
            student_gender = 0
        
        student_race = self.click_race_var.get()
        if (student_race == "Group A"):
            student_race = 0.0
        elif (student_race == "Group B"):
            student_race = 0.25
        elif (student_race == "Group C"):
            student_race = 0.50
        elif (student_race == "Group D"):
            student_race = 0.75
        else:
            student_race = 1.00

        parent_educ = self.click_parent_var.get()
        if (parent_educ == "Associate's Degree"):
            parent_educ = 0.0
        elif (parent_educ == "Bachelor's Degree"):
            parent_educ = 0.2
        elif (parent_educ == "High school"):
            parent_educ = 0.4
        elif (parent_educ == "Master's Degree"):
            parent_educ = 0.6
        elif (parent_educ == "Some College"):
            parent_educ = 0.8
        else:
            parent_educ = 1.0
        
        student_lunch = self.click_lunch_var.get()
        if (student_lunch == "Standard"):
            student_lunch = 1.0
        else:
            student_lunch = 0.0
        
        student_course = self.click_course_var.get()
        if (student_course == "Completed"):
            student_course = 0.0
        else:
            student_course = 1.0

        math_score = self.math_entry.get()
        writing_score = self.write_entry.get()
        reading_score = self.write_entry.get()

        average_score = ((math_score + writing_score + reading_score)/3)

        result_string += 'Student Performance Predicition\n'
        student_info = (student_gender, student_race, parent_educ, student_lunch, student_course, average_score)

        test_predict = best_model.predict ([student_info])
        disp_string = ("This model has a prediction accuracy of: ", str(model_accuracy))

        result = test_predict

        if(test_predict == [1]):
            result_string = (disp_string, '\n, You will pass the unit')
        else:
            result_string = (disp_string, '\n You will not pass the unit')
        
        self.test_predict_ta.insert('1.0', result_string)



my_cvd_GUI = CVD_GUI()
