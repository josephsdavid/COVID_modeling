import cotools
from cotools import text, abstract

data = cotools.Paperset('data/all')


digest = [x for x in data if 'digest' in cotools.text(x) or 'digest' in cotools.abstract(x)]

cov = ['covid', 'novel_coronavirus']
digest_covid = [x for x in digest if any(c in text(x).lower() for c in cov) or any (c in abstract(x).lower() for c in cov)]

len(digest_covid)

for d in digest_covid:
    print('-'*55)
    print('\r\n')
    print('NEW PAPER')
    print('\r\n')
    print(abstract(d))
    print(text(d))
