import urllib.request
import re
import threading
from collections import defaultdict
import matplotlib.pyplot as plt


class MapReduce:
    def __init__(self, map_func, reduce_func, num_workers=4):
        self.map_func = map_func
        self.reduce_func = reduce_func
        self.num_workers = num_workers
        self.intermediate_lock = threading.Lock()
        self.results_lock = threading.Lock()

    def execute(self, data):
        intermediate = []

        # Map phase
        map_threads = []
        for chunk in data:
            thread = threading.Thread(
                target=self.map_task, args=(chunk, intermediate))
            map_threads.append(thread)
            thread.start()

        for thread in map_threads:
            thread.join()

        # Shuffle phase: group by key
        grouped = defaultdict(list)
        for key, value in intermediate:
            grouped[key].append(value)

        # Reduce phase
        results = []
        reduce_threads = []
        for key in grouped:
            thread = threading.Thread(
                target=self.reduce_task, args=(key, grouped[key], results))
            reduce_threads.append(thread)
            thread.start()

        for thread in reduce_threads:
            thread.join()

        return sorted(results, key=lambda x: x[1], reverse=True)

    def map_task(self, chunk, intermediate):
        mapped = []
        for item in self.map_func(chunk):
            mapped.append(item)
        with self.intermediate_lock:
            intermediate.extend(mapped)

    def reduce_task(self, key, values, results):
        result = self.reduce_func(key, values)
        with self.results_lock:
            results.append(result)


def mapper(chunk):
    for word in chunk:
        yield (word, 1)


def reducer(key, values):
    return (key, sum(values))


def visualize_top_words(results, top_n=10):
    top_words = results[:top_n]
    words = [word for word, _ in top_words]
    counts = [count for _, count in top_words]
    plt.figure(figsize=(10, 6))
    plt.barh(words, counts)
    plt.xlabel('Frequency')
    plt.ylabel('Words')
    plt.title(f'Top {top_n} Most Frequent Words')
    plt.gca().invert_yaxis()
    plt.show()


if __name__ == '__main__':
    url = 'https://www.gutenberg.org/files/1342/1342-0.txt'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    response = urllib.request.urlopen(req)
    text = response.read().decode('utf-8')

    words = re.findall(r'\w+', text.lower())

    num_mappers = 4
    chunk_size = len(words) // num_mappers
    chunks = [words[i*chunk_size:(i+1)*chunk_size] for i in range(num_mappers)]
    if len(words) % num_mappers != 0:
        chunks[-1].extend(words[num_mappers*chunk_size:])

    mr = MapReduce(mapper, reducer, num_workers=num_mappers)
    results = mr.execute(chunks)

    visualize_top_words(results, 10)