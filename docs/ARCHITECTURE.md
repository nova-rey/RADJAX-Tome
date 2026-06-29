# Architecture

RADJAX-Tome is the teacher-side artifact producer. It owns corpus loading,
teacher backend invocation, artifact emission, and producer-side provenance.

Tome writes artifacts validated by RADJAX-Contract. Student code consumes those
files separately and must not be imported here.

