import html2text
class CustomHTML2Text(html2text.HTML2Text):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ignore_links = True
        self.ignore_images = True  # Set to True to skip images
        self.inside_pre = False
        self.inside_code = False

    def handle_tag(self, tag, attrs, start):
        # Skip video and iframe tags completely
        if tag in ['video', 'iframe', 'img']:
            return

        if tag == "pre":
            if start:
                self.o("```\n")
                self.inside_pre = True
            else:
                self.o("\n```")
                self.inside_pre = False
        super().handle_tag(tag, attrs, start)