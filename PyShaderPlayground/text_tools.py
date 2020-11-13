from PySide2.QtGui import QSyntaxHighlighter, QTextCharFormat, QFont
from PySide2.QtCore import Qt, QRegularExpression

class GLSLSyntaxHighlighter(QSyntaxHighlighter):
    """ Syntax highlighting for GLSL. """
    def __init__(self, parent=None):
        super().__init__(parent)

        self.highlightingRules = []

        # keywords format
        self.keyword_format = QTextCharFormat()
        self.keyword_format.setForeground(Qt.darkBlue)
        self.keyword_format.setFontWeight(QFont.Bold)
        keyword_patterns = [
            "\\bvoid\\b", "\\breturn\\b", "\\bin\\b", "\\bout\\b"
        ]
        for pattern in keyword_patterns:
            rule = HighlightingRule(pattern, self.keyword_format)
            self.highlightingRules.append(rule)
        # data types format
        self.datatype_format = QTextCharFormat()
        self.datatype_format.setForeground(Qt.darkBlue)
        datatypes_patterns = [
            "\\bfloat\\b", "\\bint\\b", "\\bbool\\b", "\\buint\\b",
            "\\bdouble\\b", "\\bvec2\\b", "\\bvec3\\b", "\\bvec4\\b",
            "\\bdvec2\\b", "\\bdvec3\\b", "\\bdvec4\\b", "\\bivec2\\b",
            "\\bivec3\\b", "\\bivec4\\b", "\\buvec2\\b", "\\buvec3\\b",
            "\\buvec4\\b", "\\bbvec2\\b", "\\bbvec3\\b", "\\bbvec4\\b",
            "\\bmat3\\b"
        ]
        for pattern in datatypes_patterns:
            rule = HighlightingRule(pattern, self.datatype_format)
            self.highlightingRules.append(rule)
        # comments format
        self.comment_format = QTextCharFormat()
        self.comment_format.setForeground(Qt.darkGray)
        rule = HighlightingRule("//[^\n]*", self.comment_format)
        self.highlightingRules.append(rule)
        self.multiline_comment_start_expr = QRegularExpression("/\\*")
        self.multiline_comment_end_expr = QRegularExpression("\\*/")
        # quotation format
        self.quotation_format = QTextCharFormat()
        self.quotation_format.setForeground(Qt.darkGreen)
        rule = HighlightingRule("\".*\"", self.quotation_format)
        self.highlightingRules.append(rule)


    def find_match(self, text: str, regexpr: QRegularExpression, offset: int = 0)
    """ Find position of a first occurence of RegExp in text. """
        found = -1
        match = regexpr.match(text, offset)
        if match.hasMatch():
            found = match.capturedStart(0)
        return found


    def highlightBlock(self, text):
        """ Main highlighting loop. """
        # simple rules
        for rule in self.highlightingRules:
            regex = QRegularExpression(rule.pattern)
            match_iterator = regex.globalMatch(text)
            while match_iterator.hasNext():
                match = match_iterator.next()
                start = match.capturedStart()
                length = match.capturedLength()
                self.setFormat(start, length, rule.rule_format)
        # multiline comments
        self.setCurrentBlockState(0)
        start_index = 0
        if self.previousBlockState() != 1:
            start_index = self.find_match(text, self.multiline_comment_start_expr, 0)
        while start_index >= 0:
            match = self.multiline_comment_end_expr.match(text, start_index)
            end_index = match.capturedStart()
            comment_length = 0
            if end_index == -1:
                self.setCurrentBlockState(1)
                comment_length = len(text) - start_index
            else:
                comment_length = end_index - start_index + match.capturedLength()
            self.setFormat(start_index, comment_length, self.comment_format)
            start_index = self.find_match(text, self.multiline_comment_start_expr, start_index + comment_length)


class HighlightingRule:
    """ Holder for pattern - format pairs. """
    def __init__(self, pattern: str, fmt: QTextCharFormat):
        self.pattern = QRegularExpression(pattern)
        self.rule_format = fmt
