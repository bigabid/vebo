import * as React from "react"
import { ChevronsUpDown, Check, Loader2 } from "lucide-react"

import { Button } from "@/components/ui/button"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { cn } from "@/lib/utils"
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command"

export interface SearchableSelectOption {
  label: string
  value: string
}

interface SearchableSelectProps {
  value: string | null
  onChange: (value: string) => void
  options: Array<string | SearchableSelectOption>
  placeholder?: string
  emptyText?: string
  disabled?: boolean
  loading?: boolean
  className?: string
}

function normalizeOptions(options: Array<string | SearchableSelectOption>): SearchableSelectOption[] {
  return options.map((opt) =>
    typeof opt === "string" ? { label: opt, value: opt } : opt
  )
}

export function SearchableSelect({
  value,
  onChange,
  options,
  placeholder = "Select...",
  emptyText = "No results found",
  disabled,
  loading,
  className,
}: SearchableSelectProps) {
  const [open, setOpen] = React.useState(false)
  const inputRef = React.useRef<HTMLInputElement | null>(null)
  const normalized = React.useMemo(() => normalizeOptions(options), [options])

  const selectedLabel = React.useMemo(() => {
    if (!value) return ""
    const found = normalized.find((o) => o.value === value)
    return found?.label ?? value
  }, [value, normalized])

  React.useEffect(() => {
    if (open) {
      const timer = setTimeout(() => {
        inputRef.current?.focus()
      }, 10)
      return () => clearTimeout(timer)
    }
  }, [open])

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          type="button"
          variant="outline"
          role="combobox"
          aria-expanded={open}
          className={cn("w-full justify-between h-12", className)}
          disabled={disabled}
        >
          <span className={cn(!value && "text-muted-foreground")}>{selectedLabel || (loading ? "Loading..." : placeholder)}</span>
          <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="p-0 w-[--radix-popover-trigger-width] min-w-[16rem]">
        <Command className="">
          <CommandInput
            ref={inputRef as any}
            placeholder="Search..."
          />
          <CommandList>
            {loading ? (
              <div className="flex items-center gap-2 px-3 py-3 text-sm text-muted-foreground">
                <Loader2 className="h-4 w-4 animate-spin" /> Loading...
              </div>
            ) : (
              <>
                <CommandEmpty>{emptyText}</CommandEmpty>
                <CommandGroup>
                  {normalized.map((opt) => (
                    <CommandItem
                      key={opt.value}
                      value={opt.label}
                      onSelect={() => {
                        onChange(opt.value)
                        setOpen(false)
                      }}
                    >
                      <Check
                        className={cn(
                          "mr-2 h-4 w-4",
                          value === opt.value ? "opacity-100" : "opacity-0"
                        )}
                      />
                      {opt.label}
                    </CommandItem>
                  ))}
                </CommandGroup>
              </>
            )}
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  )
}

interface SearchableMultiSelectProps {
  values: string[]
  onChange: (values: string[]) => void
  options: Array<string | SearchableSelectOption>
  placeholder?: string
  emptyText?: string
  disabled?: boolean
  className?: string
}

export function SearchableMultiSelect({
  values,
  onChange,
  options,
  placeholder = "Select...",
  emptyText = "No results found",
  disabled,
  className,
}: SearchableMultiSelectProps) {
  const [open, setOpen] = React.useState(false)
  const inputRef = React.useRef<HTMLInputElement | null>(null)
  const normalized = React.useMemo(() => normalizeOptions(options), [options])

  const labelSummary = React.useMemo(() => {
    if (!values?.length) return ""
    if (values.length === 1) {
      const only = normalized.find((o) => o.value === values[0])
      return only?.label ?? values[0]
    }
    return `${values.length} selected`
  }, [values, normalized])

  const toggleValue = (val: string) => {
    const set = new Set(values)
    if (set.has(val)) set.delete(val)
    else set.add(val)
    onChange(Array.from(set))
  }

  React.useEffect(() => {
    if (open) {
      const timer = setTimeout(() => {
        inputRef.current?.focus()
      }, 10)
      return () => clearTimeout(timer)
    }
  }, [open])

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          type="button"
          variant="outline"
          role="combobox"
          aria-expanded={open}
          className={cn("w-full justify-between h-10", className)}
          disabled={disabled}
        >
          <span className={cn(!values?.length && "text-muted-foreground")}>{labelSummary || placeholder}</span>
          <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="p-0 w-[--radix-popover-trigger-width] min-w-[16rem]">
        <Command>
          <CommandInput ref={inputRef as any} placeholder="Search..." />
          <CommandList>
            <CommandEmpty>{emptyText}</CommandEmpty>
            <CommandGroup>
              {normalized.map((opt) => {
                const checked = values.includes(opt.value)
                return (
                  <CommandItem
                    key={opt.value}
                    value={opt.label}
                    onSelect={() => toggleValue(opt.value)}
                  >
                    <Check className={cn("mr-2 h-4 w-4", checked ? "opacity-100" : "opacity-0")} />
                    {opt.label}
                  </CommandItem>
                )
              })}
            </CommandGroup>
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  )
}


