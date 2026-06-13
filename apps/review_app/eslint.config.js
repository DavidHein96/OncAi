// Flat ESLint config for the front-end JS in web/. Intentionally self-contained
// (no plugin dependencies) — just eslint + prettier are needed. Prettier owns
// formatting; this config only catches correctness issues.

module.exports = [
  {
    ignores: ["dist/", "build/", "node_modules/", ".venv/", "**/*.min.js"],
  },
  {
    files: ["web/**/*.js", "eslint.config.js"],
    languageOptions: {
      ecmaVersion: 2022,
      // app.js runs in the browser but exports its helpers via module.exports
      // under Node (for the test runner); app.test.js is CommonJS too.
      sourceType: "commonjs",
      globals: {
        // browser
        window: "readonly",
        document: "readonly",
        navigator: "readonly",
        alert: "readonly",
        confirm: "readonly",
        fetch: "readonly",
        setTimeout: "readonly",
        clearTimeout: "readonly",
        console: "readonly",
        // node (export shim + tests)
        module: "writable",
        require: "readonly",
      },
    },
    rules: {
      "no-undef": "error",
      "no-unused-vars": ["warn", { args: "none", caughtErrors: "none" }],
      "no-empty": ["error", { allowEmptyCatch: true }],
      "no-cond-assign": ["error", "except-parens"],
      "no-redeclare": "error",
      "no-dupe-keys": "error",
    },
  },
];
