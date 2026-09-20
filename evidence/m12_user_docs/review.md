# M12 reader review

Reviewer scope: one bounded reader-oriented pass at final content commit
4a322606b686289e36f86694c33fc77d00d5e8eb.

Initial verdict: FAIL. The reviewer reproduced that the first package example
used tar transport and then passed the tar path to mainline validate, which
returns UNSUPPORTED_ARTIFACT. This was a concrete documentation defect.

Correction: the package walkthrough now uses directory transport and validates
the unpacked directory; CLI_GUIDE explicitly says tar paths must be extracted
before validation.

Targeted recheck: PASS. Installed CLI package and directory validation returned
status=pass; M12 documentation tests passed 4/4. No runtime or semantic changes
were made. No further blocker or reservation remained.
