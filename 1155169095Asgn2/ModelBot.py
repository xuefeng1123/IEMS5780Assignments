# 1155169095 Yang Xinyi

import math
import numpy as np
import csv
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from joblib import dump, load

from telegram.ext.updater import Updater
from telegram.update import Update
from telegram.ext.callbackcontext import CallbackContext
from telegram.ext.commandhandler import CommandHandler
from telegram.ext.messagehandler import MessageHandler
from telegram.ext.filters import Filters

# 内地VPN配置
import os

os.environ["http_proxy"] = "http://127.0.0.1:41091"
os.environ["https_proxy"] = "http://127.0.0.1:41091"

# args_sample: 每个chat的参数数据
#  - model: 使用的模型编号
#  - args:  模型的输入参数列表
#  - index: 当前输入的参数，在参数列表(arg)中的序号
args_sample = {
    "index": 0,
    "model": 0,
    "args":
        [["title", ""],
         ["text", ""]]
}

# args_dic: 聊天ID-参数数据 字典
args_dic = {}

# clfs: 预加载的模型列表
clfs = [None, load('model1.pkl'), load('model2.pkl')]

updater = Updater("5152812029:AAGMs8aoOSOVoVHT54awbpIiqVIQh_a_fRI", use_context=True)


def start(update: Update, context: CallbackContext):
    update.message.reply_text(
        "welcome to real-or-fake test bot for assignment2 by Yang Xinyi 1155169095")


# CounterVectorizer
def model1(update: Update, context: CallbackContext):
    global args_dic
    setArguments(1, update.effective_chat.id)
    update.message.reply_text("What's the " + args_dic[update.effective_chat.id]['args'][0][0] + "?")


# TfidfVectorizer
def model2(update: Update, context: CallbackContext):
    global args_dic
    setArguments(2, update.effective_chat.id)
    update.message.reply_text("What's the " + args_dic[update.effective_chat.id]['args'][0][0] + "?")


# handle non-command input
def unknown(update: Update, context: CallbackContext):
    if args_dic.get(update.effective_chat.id) is None:
        setArguments(0, update.effective_chat.id)

    # arguments: 当前chat的参数数据
    arguments = args_dic[update.effective_chat.id]

    if arguments['model'] == 0:  # non-command && non-argument input
        print("invalid command")

    else:
        args_list = arguments['args']
        args_index = arguments['index']

        # when the last argument is filled:
        if args_index == len(args_list) - 1:
            args_list[args_index][1] = update.message.text
            clf = clfs[arguments['model']]
            if not (clf is None):
                out_put = clf.predict_proba([args_list[0][1] + args_list[1][1]])[0]
                print(out_put)
                if out_put[1] > 0.6:
                    texts = 'REAL'
                elif out_put[1] < 0.4:
                    texts = 'FAKE'
                else:
                    texts = 'IDK'
                update.message.reply_text("The news is " + texts + ". ( p = " + str(out_put) + " )")

                arguments['index'] = 0
                arguments['model'] = 0
                for arg in args_list:
                    arg[1] = ""
            else:
                update.message.reply_text("model loads error!")
            return

        # when there are still other not filled arguments
        if args_index < len(args_list) - 1:
            args_list[args_index][1] = update.message.text
            arguments['index'] += 1
            update.message.reply_text("What's the " + args_list[args_index + 1][0] + "?")


# set arguments data for new model using
def setArguments(model, chatId):
    global args_dic
    args_dic[chatId] = args_sample.copy()
    args_dic[chatId]['model'] = model

updater.dispatcher.add_handler(CommandHandler('start', start))
updater.dispatcher.add_handler(CommandHandler('model1', model1))
updater.dispatcher.add_handler(CommandHandler('model2', model2))

updater.dispatcher.add_handler(MessageHandler(Filters.text, unknown))
updater.dispatcher.add_handler(MessageHandler(
    Filters.command, unknown))

updater.start_polling()
