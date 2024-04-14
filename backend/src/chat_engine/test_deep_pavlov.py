from os.path import join, dirname
from time import time
from functools import reduce
import re

import nltk
from deeppavlov import configs, build_model
from deeppavlov import configs
from deeppavlov.core.commands.infer import build_model
# from deeppavlov.models.vectorizers.hashing_tfidf_vectorizer import
from deeppavlov import evaluate_model
from navec import Navec
import torch.nn as nn
from torch import Tensor
import pymorphy3

from config import MORPHY, NAVEC, PROJECT_ROOT_DIR, DEEP_PAVLOV_MODEL


def download_all():
    print("Testing deep_pavlov")
    # ranker = build_model(configs.doc_retrieval.ru_ranker_tfidf_wiki, load_trained=True)
    if DEEP_PAVLOV_MODEL:
        model = DEEP_PAVLOV_MODEL
        st = time()
        print(model(['Какая фамилия у Ленина?', 'Что такое интеграл?']))
        print(time() - st)


def deep_pavlov_test():
    print("Testing deep_pavlov")
    # ranker = build_model(configs.doc_retrieval.ru_ranker_tfidf_wiki, load_trained=True)
    if DEEP_PAVLOV_MODEL:
        print("\n"*10)
        while True:
            question=input()
            model = DEEP_PAVLOV_MODEL
            print(model([question]))



def open_question(user_question: str) -> str:
    if DEEP_PAVLOV_MODEL:
        return DEEP_PAVLOV_MODEL([user_question])[0]
    return "Модель машинного обучения для ответа на вопрос не подключена ("


def testing_map():
    while True:
        question = input()
        if test_map_drawing(question):
            print("Пользователь хочет добраться до", get_classroom_num(question))
        else:
            print("Пользователь хочет чего-то ещё")

def test_map_drawing(user_phrase: str, ):
    """
    Функция принимает на вход сообщение от пользователя,
    переводит его в вектор ембедингов и считает косинусное расстояние
    между сообщением пользователя и фразами, несущими смысл "найти какое-то место".
    Если косинусное расстояние между сообщением пользователя и хотя бы одной из фраз,
    похожих по смыслу больше определённого порога, то считаем, что пользователь запросил
    путь до какого-то объекта.
    Косинусное расстояние считается в смысловом пространстве,
    в которое нейронная сеть переводит слова, написанные на естественном языке

    :param user_phrase:
    :param proj_root_path:
    :return:
    """

    def summ(arr: str | list[str], navec: Navec):
        if isinstance(arr, str):
            arr: list[str] = arr.lower().split()
        return reduce(lambda acc, i: acc + i,
                      [navec[i] for i in arr if i in navec])

    navec_path = join(PROJECT_ROOT_DIR, 'neural', 'navec', 'navec_hudlit_v1_12B_500K_300d_100q.tar')
    navec = NAVEC.load(navec_path)

    phr = [
        'как добраться до 401 аудитории',
        'где находится 202 кабинет',
        'как попасть в четыреста пятую',
        'далеко ли триста шестая',
        'покажи путь до 17 аудитории',
    ]

    phr.append(user_phrase)

    morph = MORPHY
    m = [
        [i[0] for i in list(filter(bool, [
            [
                i1.word for i1 in morph.parse(i)
                if not ({'NUMR'} in i1.tag or {'intg'} in i1.tag or {'Anum'} in i1.tag)
            ]
            for i in i_.lower().split()
        ]))]
        for i_ in phr
    ]

    ans = [summ(i, navec) for i in m]
    ans = [nn.functional.normalize(Tensor(i), dim=0) for i in ans]
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    ans_matr: list[Tensor] = [cos(i, ans[-1]).item() for i in ans[:-1]]
    # print(ans)
    # print(*ans_matr, sep='\n')

    return max(ans_matr) > 0.45


def num2text(num: list[str]) -> str:
    dictionary = {
        'один': 1, 'два': 2, 'три': 3, 'четыре': 4, 'пять': 5, 'шесть': 6, 'семь': 7, 'восемь': 8,
        'девять': 9, 'десять': 10, 'одиннадцать': 11, 'двенадцать': 12, 'тринадцать': 13, 'четырнадцать': 14,
        'пятнадцать': 15, 'шестнадцать': 16, 'семнадцать': 17, 'восемнадцать': 18, 'девятьнадцать': 19,
        'первый': 1, 'второй': 2, 'третий': 3, 'четвёртый': 4, 'четвертый': 4, 'пятый': 5, 'шестой': 6,
        'седьмой': 7, 'восьмой': 8,
        'девятый': 9, 'десятый': 10, 'одиннадцатый': 11, 'двенадцатый': 12, 'тринадцатый': 13,
        'четырнадцатый': 14,
        'пятнадцатый': 15, 'шестнадцатый': 16, 'семнадцатый': 17, 'восемнадцатый': 18, 'девятнадцатый': 19,
        'двадцать': 20, 'тридцать': 30, 'сорок': 40, 'пятьдесят': 50, 'шестьдесят': 60, 'семьдесят': 70,
        'восемдесят': 80, 'девяносто': 90, 'сто': 100, 'двести': 200, 'триста': 300, 'четыреста': 400,
        'пятьсот': 500, 'шестьсот': 600, 'семьсот': 700, 'восемьсот': 800, 'девятьсот': 900, 'тысяча': 1000}
    try:
        # print(num)
        nums = sum([dictionary[i] for i in num if i in dictionary])
        return str(nums)
    except KeyError as exc:
        raise exc from exc
        # k = 0
        # result = ''
        # while num:
        #     num, d = divmod(num, 10)
        #     result = dictionary[d * 10 ** k] + ' ' + result
        #     k += 1
        # return result


def get_classroom_num(user_msg: str) -> str | None:
    """
    Функция выделяет место, в которое пользователь хочет попасть
    из естественного языка (сообщения пользователя)

    :param user_msg:
    :return:
    """
    morph = MORPHY
    nums = list(filter(bool, [
        [
            i1 for i1 in morph.parse(i)
            if ({'NUMR'} in i1.tag
                or {'intg'} in i1.tag
                or {'Anum'} in i1.tag
                or {'UNKN'} in i1.tag)
        ]
        for i in re.findall(r'\w+', user_msg.lower())
    ]))
    # nums = [
    #     [(i1  else i) for i1 in i]
    #     for i in nums
    # ]

    for i in nums:
        for i1 in i:
            new_nums = []
            arr = []
            if ({'NUMR'} in i1.tag or {'Anum'} in i1.tag):
                new_nums.append(i1)
            else:
                if bool(new_nums):
                    new_num = morph.parse(num2text(new_nums))
                    arr.append(new_num)
                else:
                    arr.append(i1)

    # print(*nums, sep='\n')
    unknown1 = [(i2, index) for index, i in enumerate(nums) for i2 in i
                if {'UNKN'} in i2.tag
                and bool(set('1234567890') & set(i2.word))
                and (((len(i2.word) == 2) and i2.word[0] in '1234567890' and nums[index + 1][0].word.isdigit)
                )
                ]
    unknown1 = [morph.parse(f'{unknown1[ind][0]}-{nums[unknown1[ind][1] + 1]}')[0] for ind in range(len(unknown1))]
    unknown2 = [i2 for index, i in enumerate(nums) for i2 in i
                if {'UNKN'} in i2.tag
                and bool(set('1234567890') & set(i2.word))
                and len(i2.word) > 3
                and ((i2.word[2] == '-' and i2.word[0] in '1234567890' and i2.word[3:].isdigit())
                )
                ]
    unknown = unknown2 + unknown1
    if bool(unknown):
        return unknown[0].word
    normal_intg = [i2 for i in nums for i2 in i if {'intg'} in i2.tag]
    if bool(normal_intg):
        return normal_intg[0].normal_form
    nums = [[i2.normal_form for i2 in i if not ({'intg'} in i2.tag or {'UNKN'} in i2.tag)][0]
            for i in nums]
    nums = num2text(nums)
    return nums
    # print(*nums, sep='\n')
