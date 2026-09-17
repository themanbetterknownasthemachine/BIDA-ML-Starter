#!/usr/bin/env bash
# PreToolUse-Hook: blockt gefaehrliche Operationen, egal was das Modell entscheidet.
# Eingerichtet in .claude/settings.json unter hooks.PreToolUse. Exit-Code 2 = blockieren.
# Diese Enforcement-Schicht ist fail-closed: ohne funktionierenden Python-Interpreter,
# bei nicht lesbarem Payload oder abgestuerzter Pruefung wird geblockt statt
# stillschweigend durchgelassen.

py_bin=""
for c in python3 python py; do
  if "$c" -c "" >/dev/null 2>&1; then
    py_bin="$c"
    break
  fi
done
if [ -z "$py_bin" ]; then
  echo "Blockiert durch protect-files.sh: kein Python-Interpreter gefunden, Pruefung nicht moeglich." >&2
  exit 2
fi

reason="$("$py_bin" "$(dirname "$0")/_check.py")"
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "Blockiert durch protect-files.sh: Pruefung abgebrochen (Exit ${rc}), Aktion nicht freigegeben." >&2
  exit 2
fi
if [ -n "$reason" ]; then
  echo "Blockiert durch protect-files.sh: ${reason} ist ohne explizite Freigabe gesperrt." >&2
  exit 2
fi
exit 0
