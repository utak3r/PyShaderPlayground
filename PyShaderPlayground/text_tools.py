from PySide6.QtGui import QSyntaxHighlighter, QTextCharFormat, QFont
from PySide6.QtCore import Qt, QRegularExpression

class GLSLSyntaxHighlighter(QSyntaxHighlighter):
    """ Syntax highlighting for GLSL. """

    THEMES = {
        'light': {
            'keyword': Qt.darkBlue,
            'datatype': Qt.darkBlue,
            'comment': Qt.darkGray,
            'quotation': Qt.darkGreen,
            'bold_keywords': True
        },
        'dark': {
            'keyword': Qt.cyan,
            'datatype': Qt.magenta,
            'comment': Qt.gray,
            'quotation': Qt.yellow,
            'bold_keywords': True
        }
    }

    def __init__(self, parent=None):
        super().__init__(parent)

        self.highlightingRules = []

        # Initialize formats
        self.keyword_format = QTextCharFormat()
        self.datatype_format = QTextCharFormat()
        self.comment_format = QTextCharFormat()
        self.quotation_format = QTextCharFormat()

        # Define patterns
        keyword_patterns = ["\\bvoid\\b", "\\breturn\\b", "\\bin\\b", "\\bout\\b"]
        datatypes_patterns = [
            "\\bfloat\\b", "\\bint\\b", "\\bbool\\b", "\\buint\\b",
            "\\bdouble\\b", "\\bvec2\\b", "\\bvec3\\b", "\\bvec4\\b",
            "\\bdvec2\\b", "\\bdvec3\\b", "\\bdvec4\\b", "\\bivec2\\b",
            "\\bivec3\\b", "\\bivec4\\b", "\\buvec2\\b", "\\buvec3\\b",
            "\\buvec4\\b", "\\bbvec2\\b", "\\bbvec3\\b", "\\bbvec4\\b",
            "\\bmat3\\b"
        ]

        # Create rules
        for pattern in keyword_patterns:
            self.highlightingRules.append(HighlightingRule(pattern, self.keyword_format))
        for pattern in datatypes_patterns:
            self.highlightingRules.append(HighlightingRule(pattern, self.datatype_format))
        
        self.highlightingRules.append(HighlightingRule("//[^\n]*", self.comment_format))
        self.highlightingRules.append(HighlightingRule("\".*\"", self.quotation_format))

        self.multiline_comment_start_expr = QRegularExpression("/\\*")
        self.multiline_comment_end_expr = QRegularExpression("\\*/")

        # Set default theme
        self.set_theme('light')

    def set_theme(self, theme_name: str):
        """ Change the colors depending on theme (light/dark). """
        if theme_name not in self.THEMES:
            return

        theme = self.THEMES[theme_name]
        
        self.keyword_format.setForeground(theme['keyword'])
        if theme.get('bold_keywords', False):
            self.keyword_format.setFontWeight(QFont.Bold)
        else:
            self.keyword_format.setFontWeight(QFont.Normal)

        self.datatype_format.setForeground(theme['datatype'])
        self.comment_format.setForeground(theme['comment'])
        self.quotation_format.setForeground(theme['quotation'])

        self.rehighlight()

    @staticmethod
    def find_match(text: str, regexpr: QRegularExpression, offset: int = 0) -> int:
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
