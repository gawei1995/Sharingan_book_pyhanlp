class Node():
    def __init__(self,value) -> None:
        self._children = {}
        self._value = value

    def _add_child(self,char,value,overwrite=False):
        child = self._children.get(char)
        if child is None:
            child = Node(value)
            self._children[char] = child
        elif overwrite:
            child._value = value
        return child

class Trie(Node):
    def __init__(self) -> None:
        super(Trie, self).__init__(None)

    def __contains__(self, key):
        return self[key] is not None

    def __getitem__(self, key):
        state = self

        for char in key:
            state = state._children.get(char)
            if state is None:
                return None
        return state._value

    def __setitem__(self, key, value):
        state = self
        for i,char in enumerate(key):
            if i < len(key) -1:
                state = state._add_child(char,None,False)
            else:
                state = state._add_child(char,value,True)


if __name__ == '__main__':
    trie = Trie()
    trie['自然'] = 'nature'
    trie['自然人'] = 'human'
    trie['自然语言'] = 'language'
    trie['自语'] = 'talk to oneself'
    trie['入门'] = 'introduction'
    assert '自然' in trie
    trie['自然'] = None
    assert '自然' in trie

    trie['自然语言'] = 'human language'