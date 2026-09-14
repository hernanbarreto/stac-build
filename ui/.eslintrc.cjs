module.exports = {
  root: true,
  env: { browser: true, es2020: true },
  extends: [
    'eslint:recommended',
    'plugin:@typescript-eslint/recommended',
    'plugin:react-hooks/recommended',
  ],
  ignorePatterns: ['dist', 'dist-electron', 'release', '.eslintrc.cjs', 'scripts'],
  parser: '@typescript-eslint/parser',
  plugins: ['react-refresh'],
  rules: {
    // Backend payloads (acta, ledger, reports) are open JSON documents that the
    // UI renders as received; typing them as `any` at the boundary is deliberate.
    '@typescript-eslint/no-explicit-any': 'off',
    // Contexts and hook modules export their provider AND their hooks (i18n,
    // layout, toasts); splitting them only to satisfy fast-refresh adds files
    // without value.
    'react-refresh/only-export-components': 'off',
    // The viewport wires Three.js listeners once and reads live values through
    // refs on purpose (re-subscribing on every render broke the render loop);
    // the remaining omissions are documented inline where they matter.
    'react-hooks/exhaustive-deps': 'off',
    '@typescript-eslint/no-unused-vars': ['error', { argsIgnorePattern: '^_', varsIgnorePattern: '^_' }],
  },
}
