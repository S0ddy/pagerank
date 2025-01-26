import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """

    # create dictionary
    transition = dict()

    # if page has links
    links = corpus.get(page)

    if links:
        for link in links:
            # devide the damping_factor equally across all children 
            transition[link] = damping_factor/len(links) 
        for key in corpus: 
            # add (1-damping_factor) equally across all keys 
            if key in transition: 
                transition[key] = transition[key] + ((1-damping_factor)/len(corpus))
            else:
                transition[key] = (1-damping_factor)/len(corpus)
    else:
        for key in corpus:
            # add chances equally across all keys
            transition[key] = 1/len(corpus)
    
    return transition
    

def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # get a random page from corpus
    page = random.choice(list(corpus.keys()))
    clicks = dict()
    clicks[page] = clicks.get(page, 0) + 1

    # get my first transition
    transition = transition_model(corpus, page, damping_factor)

    for i in range(n):
        # go into the next transition based on the previous transition
        page = random.choices(list(transition.keys()), list(transition.values()))[0]
        clicks[page] = clicks.get(page, 0) + 1
        transition = transition_model(corpus, page, damping_factor)

    # the sum of all clicks should be 1
    amount_of_clicks = sum(clicks.values())
    
    for key, value in clicks.items():
        clicks[key] = value/amount_of_clicks
    
    return clicks


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    
    result = dict()

    assign_initial_rank(corpus, result)

    return recursive_page_rank(corpus, result, damping_factor)


def assign_initial_rank(corpus, result):
    num_of_pages = len(corpus.values())
    init_page_rank = 1 / num_of_pages

    for key in corpus.keys():
        result[key] = init_page_rank


def recursive_page_rank(corpus, result, damping_factor):
    num_of_pages = len(corpus.keys())
    chance_of_random_page = (1 - damping_factor) / num_of_pages
    
    # A page that has no links at all should be interpreted as having one link for 
    # every page in the corpus (including itself).
    for key, values in corpus.items():
        if not values:
            corpus[key] = set(corpus.keys())

    return update_page_rank(corpus, result, damping_factor, chance_of_random_page)


def update_page_rank(corpus, result, damping_factor, chance_of_random_page):
    new_result = dict()
    repeat = False

    # iterate over each page and update rank
    for page in corpus.keys():
        sum_from_formula = 0
        links = set()
        # list of keys that points to our page
        for key, values in corpus.items():
            for value in values:
                if value == page:
                    links.add(key)
        
        for link in links:
            sum_from_formula += result.get(link) / len(corpus.get(link))

        new_rank = chance_of_random_page + damping_factor * sum_from_formula

        new_result[page] = new_rank
    
    for key in new_result.keys():
        if abs(new_result[key] - result[key]) > 0.001:
            repeat = True
    
    result = new_result

    if repeat:
        result = update_page_rank(corpus, result, damping_factor, chance_of_random_page)

    return result


if __name__ == "__main__":
    main()
