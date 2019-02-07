class Chunck:
    def __init__(self,type,b_idx,length):
        self.type=type
        self.b_idx=b_idx
        self.length=length

    def __str__(self):
        return f'({self.type},{self.b_idx},{self.length})'

    def equals_total(self,chunck):
        return self.type==chunck.type and self.b_idx==chunck.b_idx and self.length==chunck.length

    def equals_name(self,chunck):
        return  self.b_idx==chunck.b_idx and self.length==chunck.length

    def __repr__(self):
        return f'({self.type},{self.b_idx},{self.length})'

