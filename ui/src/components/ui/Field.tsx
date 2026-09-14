/**
 * Form controls (§6): Input, NumberInput (fixed unit suffix), Select,
 * Slider (value in mono), SegmentedControl, Checkbox, Switch, Field label.
 * Heights follow the density; every control has hover / focus-visible /
 * disabled states from the tokens. Nothing here decides a colour.
 */
import { forwardRef, useId, type InputHTMLAttributes, type ReactNode, type SelectHTMLAttributes } from 'react'
import { Check, ChevronDown, Minus, Search, X } from 'lucide-react'
import { useFmt, useT } from '../../i18n'
import type { ControlSize } from './Button'

// ── Field wrapper ───────────────────────────────────────────────────────
export function Field({ label, hint, error, inline = false, children, htmlFor }: { label?: ReactNode; hint?: ReactNode; error?: ReactNode; inline?: boolean; children: ReactNode; htmlFor?: string }) {
  return (
    <div className={`stac-field ${inline ? 'stac-field--inline' : ''} ${error ? 'stac-field--error' : ''}`.trim()}>
      {label && <label className="stac-field__label" htmlFor={htmlFor}>{label}</label>}
      <div className="stac-field__control">{children}</div>
      {hint && !error && <div className="stac-field__hint">{hint}</div>}
      {error && <div className="stac-field__error" role="alert">{error}</div>}
    </div>
  )
}

// ── Input ───────────────────────────────────────────────────────────────
export interface InputProps extends Omit<InputHTMLAttributes<HTMLInputElement>, 'size'> {
  size?: ControlSize
  icon?: ReactNode
  onClear?: () => void
  mono?: boolean
  invalid?: boolean
}

export const Input = forwardRef<HTMLInputElement, InputProps>(function Input({ size = 'md', icon, onClear, mono = false, invalid = false, className = '', value, ...rest }, ref) {
  const t = useT()
  return (
    <span className={`stac-input stac-input--${size} ${icon ? 'stac-input--icon' : ''} ${invalid ? 'stac-input--invalid' : ''} ${className}`.trim()}>
      {icon && <span className="stac-input__icon" aria-hidden>{icon}</span>}
      <input ref={ref} className={`stac-input__el ${mono ? 'stac-mono' : ''}`.trim()} value={value} aria-invalid={invalid || undefined} {...rest} />
      {onClear && value != null && String(value) !== '' && (
        <button type="button" className="stac-input__clear" aria-label={t('common.clear')} onClick={onClear} tabIndex={-1}><X aria-hidden /></button>
      )}
    </span>
  )
})

export const SearchInput = forwardRef<HTMLInputElement, InputProps>(function SearchInput(props, ref) {
  return <Input ref={ref} type="search" icon={<Search aria-hidden />} {...props} />
})

// ── NumberInput with a fixed unit suffix ────────────────────────────────
export interface NumberInputProps extends Omit<InputHTMLAttributes<HTMLInputElement>, 'size' | 'onChange' | 'value'> {
  value: number | ''
  onChange: (v: number | '') => void
  unit?: string
  size?: ControlSize
  digits?: number
}

export const NumberInput = forwardRef<HTMLInputElement, NumberInputProps>(function NumberInput({ value, onChange, unit, size = 'md', className = '', ...rest }, ref) {
  return (
    <span className={`stac-input stac-input--${size} stac-input--number ${className}`.trim()}>
      <input ref={ref} type="number" inputMode="decimal" className="stac-input__el stac-mono" value={value}
        onChange={e => onChange(e.target.value === '' ? '' : Number(e.target.value))} {...rest} />
      {unit && <span className="stac-input__unit" aria-hidden>{unit}</span>}
    </span>
  )
})

// ── Select ──────────────────────────────────────────────────────────────
export interface SelectOption<T extends string = string> { value: T; label: string; disabled?: boolean }
export interface SelectProps<T extends string> extends Omit<SelectHTMLAttributes<HTMLSelectElement>, 'size' | 'onChange' | 'value'> {
  value: T | ''
  onChange: (v: T) => void
  options: SelectOption<T>[]
  placeholder?: string
  size?: ControlSize
}

export function Select<T extends string>({ value, onChange, options, placeholder, size = 'md', className = '', ...rest }: SelectProps<T>) {
  return (
    <span className={`stac-select stac-select--${size} ${className}`.trim()}>
      <select className="stac-select__el" value={value} onChange={e => onChange(e.target.value as T)} {...rest}>
        {placeholder != null && <option value="">{placeholder}</option>}
        {options.map(o => <option key={o.value} value={o.value} disabled={o.disabled}>{o.label}</option>)}
      </select>
      <ChevronDown className="stac-select__chevron" aria-hidden />
    </span>
  )
}

// ── Slider with a mono readout ──────────────────────────────────────────
export interface SliderProps {
  value: number
  onChange: (v: number) => void
  min: number
  max: number
  step?: number
  label?: ReactNode
  format?: (v: number) => string
  unit?: string
  digits?: number
  tone?: 'brand' | 'measure' | 'err'
  disabled?: boolean
  className?: string
  ariaLabel?: string
}

export function Slider({ value, onChange, min, max, step = 1, label, format, unit, digits = 0, tone = 'brand', disabled, className = '', ariaLabel }: SliderProps) {
  const fmt = useFmt()
  const id = useId()
  const text = format ? format(value) : `${fmt.number(value, digits)}${unit ? ` ${unit}` : ''}`
  return (
    <div className={`stac-slider stac-slider--${tone} ${className}`.trim()}>
      {label && <label className="stac-slider__label" htmlFor={id}>{label}</label>}
      <input id={id} type="range" className="stac-slider__el" min={min} max={max} step={step} value={value} disabled={disabled}
        aria-label={ariaLabel} onChange={e => onChange(parseFloat(e.target.value))} />
      <output className="stac-slider__value" htmlFor={id}>{text}</output>
    </div>
  )
}

// ── SegmentedControl ────────────────────────────────────────────────────
export interface SegmentOption<T extends string> { value: T; label?: ReactNode; icon?: ReactNode; title?: string; disabled?: boolean }

export function SegmentedControl<T extends string>({ value, onChange, options, size = 'md', ariaLabel, className = '' }: { value: T; onChange: (v: T) => void; options: SegmentOption<T>[]; size?: ControlSize; ariaLabel: string; className?: string }) {
  return (
    <div role="radiogroup" aria-label={ariaLabel} className={`stac-segmented stac-segmented--${size} ${className}`.trim()}>
      {options.map(o => (
        <button key={o.value} type="button" role="radio" aria-checked={value === o.value} title={o.title} disabled={o.disabled}
          className={`stac-segmented__item ${value === o.value ? 'stac-segmented__item--on' : ''} ${!o.label ? 'stac-segmented__item--icon' : ''}`.trim()}
          onClick={() => onChange(o.value)}>
          {o.icon && <span className="stac-segmented__icon" aria-hidden>{o.icon}</span>}
          {o.label && <span>{o.label}</span>}
        </button>
      ))}
    </div>
  )
}

// ── Checkbox ────────────────────────────────────────────────────────────
export interface CheckboxProps extends Omit<InputHTMLAttributes<HTMLInputElement>, 'type' | 'onChange'> {
  checked: boolean
  indeterminate?: boolean
  onChange: (checked: boolean) => void
  label?: ReactNode
  description?: ReactNode
}

export const Checkbox = forwardRef<HTMLInputElement, CheckboxProps>(function Checkbox({ checked, indeterminate = false, onChange, label, description, className = '', disabled, ...rest }, ref) {
  return (
    <label className={`stac-check ${disabled ? 'stac-check--disabled' : ''} ${className}`.trim()}>
      <input ref={ref} type="checkbox" className="stac-check__input" checked={checked} disabled={disabled} onChange={e => onChange(e.target.checked)} {...rest} />
      <span className={`stac-check__box ${indeterminate ? 'stac-check__box--mixed' : ''}`.trim()} aria-hidden>
        {indeterminate ? <Minus /> : checked ? <Check /> : null}
      </span>
      {label != null && <span className="stac-check__label">{label}{description && <span className="stac-check__desc">{description}</span>}</span>}
    </label>
  )
})

// ── Switch ──────────────────────────────────────────────────────────────
export function Switch({ checked, onChange, label, disabled, className = '', ariaLabel }: { checked: boolean; onChange: (v: boolean) => void; label?: ReactNode; disabled?: boolean; className?: string; ariaLabel?: string }) {
  return (
    <label className={`stac-switch ${disabled ? 'stac-switch--disabled' : ''} ${className}`.trim()}>
      <input type="checkbox" role="switch" className="stac-switch__input" checked={checked} disabled={disabled} aria-label={ariaLabel} onChange={e => onChange(e.target.checked)} />
      <span className="stac-switch__track" aria-hidden><span className="stac-switch__knob" /></span>
      {label != null && <span className="stac-switch__label">{label}</span>}
    </label>
  )
}
