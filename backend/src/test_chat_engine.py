import os

from chat_engine.test_deep_pavlov import deep_pavlov_test, test_map_drawing, get_classroom_num, download_all, testing_map

if __name__ == '__main__':

    ROOT_DIR = os.path.split( os.path.dirname(__file__))[0]
    PROJECT_ROOT_DIR = os.environ.get("PROJECT_ROOT_DIR", None) or os.path.dirname(ROOT_DIR)
    print(PROJECT_ROOT_DIR)
    # testing_map()
    # download_all()
    deep_pavlov_test()
    # r1 = test_map_drawing("Нарисуй карту до 71 аудитории")
    # r2 = test_map_drawing("Я люблю рисовать")
    # print(r1, r2)
    # res = get_classroom_num("Нарисуй карту до аудитории в четыреста пятую")
    # print(res)
