from cacheguard.data.newsgroups import clean_text, load_newsgroups

RAW_POST = """Xref: cantaloupe.srv.cs.cmu.edu sci.space:59497 sci.astro:33665
Path: cantaloupe.srv.cs.cmu.edu!crabapple.srv.cs.cmu.edu!news
Newsgroups: sci.space,sci.astro
From: et@teal.csn.org (Eric H. Taylor)
Subject: Re: Gravity waves
Lines: 9

In article <C4KvJF.4qo@well.sf.ca.us> metares@well.sf.ca.us (Tom) writes:
>Existence is undefined unless it is observable.
> More quoted text here.
Gravity waves, if they exist, should be measurable. See http://example.com/paper
Email me at someone@example.com for details.
--
Eric H. Taylor | 4-L Laboratories
"""


def test_headers_are_removed_so_the_label_does_not_leak():
    cleaned = clean_text(RAW_POST)
    for leaked in ("Xref", "Path", "Newsgroups", "sci.space", "cantaloupe", "Subject"):
        assert leaked not in cleaned


def test_quotes_signature_emails_and_urls_are_removed():
    cleaned = clean_text(RAW_POST)
    assert "Existence is undefined" not in cleaned
    assert "In article" not in cleaned
    assert "4-L Laboratories" not in cleaned
    assert "@" not in cleaned
    assert "http" not in cleaned


def test_body_text_and_punctuation_are_kept():
    cleaned = clean_text(RAW_POST)
    assert cleaned.startswith("Gravity waves, if they exist, should be measurable.")


def test_post_without_header_separator_is_empty():
    assert clean_text("Subject: only headers\nFrom: x") == ""


def _write(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("Subject: s\nFrom: f\n\n" + body, encoding="latin-1")


def test_loader_labels_from_folders_skips_stray_files_and_dedupes(tmp_path):
    _write(tmp_path / "sci.space" / "1", "Rockets need a lot of fuel to reach orbit.")
    _write(tmp_path / "sci.space" / "2", "short")
    _write(tmp_path / "talk.politics" / "3", "Rockets need a lot of fuel to reach orbit.")  # cross-post
    _write(tmp_path / "talk.politics" / "4", "Taxes were the main topic of the debate.")
    (tmp_path / "README.txt").write_text("not a newsgroup")

    documents, stats = load_newsgroups(tmp_path)

    assert [(d.doc_id, d.newsgroup) for d in documents] == [(0, "sci.space"), (1, "talk.politics")]
    assert stats.files_read == 4
    assert stats.duplicates == 1
    assert stats.too_short == 1
    assert stats.per_newsgroup == {"sci.space": 1, "talk.politics": 1}
