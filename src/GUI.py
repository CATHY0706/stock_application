#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 14:35:27 2020

Graphical User Interface
@author: Tianyi Zhang (19202673)

This program does not implement all the functions in the stock quotes application,
only implements the function of plotting Moving Average Convergence Divergence graph.

"""

from tkinter import *
from tkinter import messagebox
from StockQuotesApplication import *


class Application(Frame):

    def __init__(self, root):
        """
        Create a main window
        :param root: Tk()
        """
        super().__init__(root)
        self.master = root
        self.pack()

        # variables get from user
        self.symbol = StringVar()
        self.start_date = StringVar()
        self.end_date = StringVar()

        root.title('Tianyi\'s Stock Application')
        # Size: width:430 height:230, Location: x:400 y:200
        root.geometry('430x230+400+200')
        # minimum size of GUI
        root.minsize(430, 230)

        # call function createWidget()
        self.createWidget()

    def createWidget(self):
        """
        Create Widgets
        """
        # welcome
        frame1 = Frame(self)
        label01 = Label(frame1, borderwidth=0, relief='solid', justify='center')
        label01[
            'text'] = 'Welcome to Tianyi\'s Stock Application!\n(Only show the Moving Average Convergence Divergence chart)'
        label01.grid(row=0, column=1)

        # label and entry for symbol
        frame2 = Frame(self)
        Label(frame2, text='Stock Symbol: ', width=25).grid(row=0, column=0)
        Entry(frame2, width=25, textvariable=self.symbol).grid(row=0, column=1)

        # label and entry for start date
        frame3 = Frame(self)
        Label(frame3, text='Start Date (yyyy-mm-dd): ', width=25).grid(row=0, column=0)
        Entry(frame3, width=25, textvariable=self.start_date).grid(row=0, column=1)

        # label and entry for end date
        frame4 = Frame(self)
        Label(frame4, text='End Date (yyyy-mm-dd): ', width=25).grid(row=0, column=0)
        Entry(frame4, width=25, textvariable=self.end_date).grid(row=0, column=1)

        # submit and quit buttons
        frame5 = Frame(self)
        # if submit, call function descriptive()
        Button(frame5, text='Submit', width=12, command=self.descriptive).grid(row=0, column=0, padx=15)
        # if quit, end the program
        Button(frame5, text='Quit', width=12, command=root.destroy).grid(row=0, column=1, padx=15)

        # Assemble and organize all the frames
        frame1.grid(pady=10)
        frame2.grid(pady=5)
        frame3.grid(pady=5)
        frame4.grid(pady=5)
        frame5.grid(pady=12)

    def descriptive(self):
        """
        get values entered by user and plot the Moving Average Convergence Divergence chart
        """
        # get values entered by user
        symbol = self.symbol.get()
        start = self.start_date.get()
        end = self.end_date.get()

        # If one of the values is empty, then a warning statement will be returned, requiring complete information
        if len(symbol) == 0 or len(start) == 0 or len(end) == 0:
            messagebox.showinfo('Warning', 'Please complete the information!')
            # stop the conditional execution by return
            return

        # If all the values are entered, start plotting
        else:
            # to return the values entered by user
            messagebox.showinfo('Information', 'What you entered is:\n Symbol: %s\n Start date: %s\n End date: %s' % (
                symbol, start, end))

            # call functions in StockQuotes and plot the Moving Average Convergence Divergence chart
            historical_quotes = get_all_historical_quotes(symbol.upper())
            historical_quotes_selected = query_time_series(historical_quotes, start, end)
            show_data(historical_quotes_selected, start, end, symbol)
            MACD(historical_quotes_selected, start, end, symbol)


if __name__ == '__main__':
    root = Tk()
    application = Application(root=root)
    application.mainloop()
